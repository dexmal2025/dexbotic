import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import torch
import torch.distributed as dist
from codetiming import Timer
from loguru import logger
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

from dexbotic.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from dexbotic.data.dataset.dex_rl_dataset import BufferedRLDataLoader
from dexbotic.exp.base_exp import BaseExp, DataConfig, TokenizerConfig
from dexbotic.exp.rl.rl_base import GRPOConfig, RLEnvironmentConfig
from dexbotic.exp.rl.rl_rollout_redis import redistribute_filtered_batch_circular
from dexbotic.exp.rl.rl_trainer import (
    DexboticRLTrainer,
    FixedKLController,
    RLTrainerConfig,
    RobRewardManager,
    apply_kl_penalty,
    compute_advantage,
    masked_mean,
)
from dexbotic.exp.rl.rl_utils import quat2axisangle, read_normalization_stats
from dexbotic.sim_envs.libero.libero_utils import (
    get_libero_image,
    get_libero_wrist_image,
)
from dexbotic.tokenization import conversation as conversation_lib
from dexbotic.tokenization.tokenization import tokenizer_image_token

from .oft_discrete_exp import OFTDiscreteModelConfig


def reduce_metrics(metrics: dict):
    """Reduce metrics by computing mean of lists"""
    for key, val in metrics.items():
        if isinstance(val, list):
            metrics[key] = np.mean(val)
    return metrics


def compute_data_metrics(batch: Dict[str, Any], config) -> Dict[str, float]:
    """
    Compute data metrics from batch

    Args:
        batch: Dictionary containing batch data with keys:
            - token_level_scores: Tensor of shape (batch_size, seq_len)
            - token_level_rewards: Tensor of shape (batch_size, seq_len)
            - advantages: Tensor of shape (batch_size, seq_len)
            - returns: Tensor of shape (batch_size, seq_len)
            - finish_step: Tensor of shape (batch_size,)
            - responses: Tensor of shape (batch_size, traj_len, chunk_size)
        config: Configuration object

    Returns:
        Dictionary of metrics
    """
    try:
        sequence_score = batch["token_level_scores"].sum(-1)
        sequence_reward = batch["token_level_rewards"].sum(-1)
        advantages = batch["advantages"]
        returns = batch["returns"]

        # Compute response mask
        finish_step = batch["finish_step"] * config.model_config.model.action_dim
        responses = batch["responses"]
        traj_length = responses.size(1) * responses.size(2)

        steps = torch.arange(traj_length, device=advantages.device)
        steps_expanded = steps.unsqueeze(0).expand(responses.size(0), -1)
        response_mask = steps_expanded < finish_step.unsqueeze(1)

        # Check if we have valid data
        if response_mask.sum() == 0:
            return {
                "critic/score/mean": 0.0,
                "critic/rewards/mean": 0.0,
                "critic/advantages/mean": 0.0,
                "critic/returns/mean": 0.0,
            }

        # Compute metrics with minimal memory usage
        valid_advantages = advantages[response_mask.bool()]
        valid_returns = returns[response_mask.bool()]

        # Compute masked means
        adv_mean = masked_mean(advantages, response_mask)
        ret_mean = masked_mean(returns, response_mask)

        # Check for NaN/inf values and replace with 0
        adv_mean_item = (
            adv_mean.detach().item()
            if not torch.isnan(adv_mean) and not torch.isinf(adv_mean)
            else 0.0
        )
        ret_mean_item = (
            ret_mean.detach().item()
            if not torch.isnan(ret_mean) and not torch.isinf(ret_mean)
            else 0.0
        )

        metrics = {
            # Only keep essential metrics
            "critic/score/mean": torch.mean(sequence_score).detach().item(),
            "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
            "critic/advantages/mean": adv_mean_item,
            "critic/returns/mean": ret_mean_item,
        }

        # Clean up intermediate tensors
        del valid_advantages, valid_returns, adv_mean, ret_mean
        del steps, steps_expanded, response_mask

    except Exception:
        # Return minimal default metrics
        metrics = {
            "critic/score/mean": 0.0,
            "critic/rewards/mean": 0.0,
            "critic/advantages/mean": 0.0,
            "critic/returns/mean": 0.0,
        }
    return metrics


@dataclass
class ModelSubConfig(OFTDiscreteModelConfig):
    """Model-specific configuration for actor-rollout-ref architecture"""

    model_name_or_path: str = field(default="")
    action_model_type: str = field(default="Discrete")
    action_dim: int = field(default=7)  # Action token length
    chunk_size: int = field(default=8)  # Action chunks length
    num_bins: int = field(
        default=256
    )  # Number of discrete bins for action quantization
    use_proprio: bool = field(default=False)


@dataclass
class OptimConfig:
    """Optimizer configuration"""

    lr: float = field(default=5e-6)  # Learning rate
    warmup_style: str = field(default="constant")  # Warmup style


@dataclass
class FSDPConfig:
    """FSDP configuration for distributed training"""

    param_offload: bool = field(default=False)
    grad_offload: bool = field(default=True)
    optimizer_offload: bool = field(default=True)


@dataclass
class ActorSubConfig:
    """Actor-specific configuration"""

    optim: OptimConfig = field(default_factory=OptimConfig)
    ppo_mini_batch_size: int = field(default=128)
    ppo_micro_batch_size: int = field(default=8)
    use_dynamic_bsz: bool = field(default=False)
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    grad_clip: float = field(default=1.0)
    clip_ratio_high: float = field(default=0.28)
    clip_ratio_low: float = field(default=0.2)
    num_images_in_input: int = field(default=1)
    traj_mini_batch_size: int = field(default=4)  # default 16
    entropy_coeff: float = field(default=0.0)


@dataclass
class RolloutSubConfig:
    """Rollout-specific configuration"""

    num_images_in_input: int = field(default=1)
    use_proprio: bool = field(default=False)
    temperature: float = field(default=1.6)
    micro_batch_size: int = field(default=1)
    unnorm_key: str = field(default="libero_10")
    model_family: str = field(default="openvla")
    task_suite_name: str = field(default="libero_10")
    num_steps_wait: int = field(default=10)
    log_prob_micro_batch_size: int = field(default=32)


@dataclass
class RefSubConfig:
    """Reference model configuration"""

    log_prob_micro_batch_size: int = field(default=32)
    fsdp_config: FSDPConfig = field(
        default_factory=lambda: FSDPConfig(param_offload=True)
    )


@dataclass
class SimpleVLAActorRolloutRefConfig:
    """
    Complete actor-rollout-ref configuration
    This replaces SimpleVLARLModelConfig with a more comprehensive structure

    Provides property delegation for backward compatibility with code expecting
    direct access to model attributes (e.g., config.model_name_or_path instead
    of config.model.model_name_or_path)
    """

    model: ModelSubConfig = field(default_factory=ModelSubConfig)
    actor: ActorSubConfig = field(default_factory=ActorSubConfig)
    rollout: RolloutSubConfig = field(default_factory=RolloutSubConfig)
    ref: RefSubConfig = field(default_factory=RefSubConfig)

    def build_model(self):
        return self.model.build_model()

    # ========== Property delegation for backward compatibility ==========
    # These properties allow direct access like config.model_name_or_path
    # instead of config.model.model_name_or_path

    @property
    def model_name_or_path(self) -> str:
        """Delegate to model.model_name_or_path"""
        return self.model.model_name_or_path

    @model_name_or_path.setter
    def model_name_or_path(self, value: str):
        """Delegate to model.model_name_or_path"""
        self.model.model_name_or_path = value

    @property
    def chat_template(self) -> str:
        """Delegate to model.chat_template"""
        return self.model.chat_template

    @chat_template.setter
    def chat_template(self, value: str):
        """Delegate to model.chat_template"""
        self.model.chat_template = value

    @property
    def action_dim(self) -> int:
        """Delegate to model.action_dim"""
        return self.model.action_dim

    @action_dim.setter
    def action_dim(self, value: int):
        """Delegate to model.action_dim"""
        self.model.action_dim = value

    @property
    def chunk_size(self) -> int:
        """Delegate to model.chunk_size"""
        return self.model.chunk_size

    @chunk_size.setter
    def chunk_size(self, value: int):
        """Delegate to model.chunk_size"""
        self.model.chunk_size = value

    @property
    def action_model_type(self) -> str:
        """Delegate to model.action_model_type"""
        return self.model.action_model_type

    @action_model_type.setter
    def action_model_type(self, value: str):
        """Delegate to model.action_model_type"""
        self.model.action_model_type = value

    @property
    def num_bins(self) -> int:
        """Delegate to model.num_bins"""
        return self.model.num_bins

    @num_bins.setter
    def num_bins(self, value: int):
        """Delegate to model.num_bins"""
        self.model.num_bins = value

    @property
    def use_proprio(self) -> bool:
        """Delegate to model.use_proprio"""
        return self.model.use_proprio

    @use_proprio.setter
    def use_proprio(self, value: bool):
        """Delegate to model.use_proprio"""
        self.model.use_proprio = value


@dataclass
class SimpleVLARLDataConfig(DataConfig):
    """
    Data configuration for SimpleVLA-RL
    """

    auto_norm: bool = field(default=False)

    # Environment type and task configuration
    env_type: str = field(default="libero")
    task_name: str = field(default="libero_10")  # Specific task name
    num_trials_per_task: int = field(default=50)

    # RL dataset parameters
    batch_size: int = field(
        default=8
    )  # Number of base environments per batch -- default=8
    n_sample: int = field(
        default=8
    )  # Number of samples for GRPO interleaving (applied in dataloader) --  default=8
    target_rollouts_num: int = field(default=32)  # Target number of rollouts per gpu
    # Training/validation split
    train_val: str = field(default="train")  # "train" or "valid"

    # Dataset control parameters
    shuffle: bool = field(default=True)  # Whether to shuffle the dataset
    seed: int = field(default=42)  # Random seed for reproducibility
    drop_last: bool = field(default=False)  # Whether to drop the last incomplete batch

    # Data filtering
    filter_accuracy: bool = field(default=False)
    filter_truncated: bool = field(default=False)
    accuracy_lower_bound: float = field(default=0.1)
    accuracy_upper_bound: float = field(default=0.9)
    oversample_factor: int = field(default=1)

    # Batch sizes for training
    train_batch_size: int = field(default=64)
    val_batch_size: int = field(default=496)

    # Sequence lengths
    max_prompt_length: int = field(default=128)
    max_response_length: int = field(default=128)

    # Vision input
    num_images_in_input: int = field(default=1)
    use_proprio: bool = field(default=False)


@dataclass
class SimpleVLARLTrainerConfig(RLTrainerConfig):
    """
    Trainer configuration for SimpleVLA-RL
    """

    # Learning rates
    actor_lr: float = field(default=5e-6)
    warmup_style: str = field(default="constant")

    # PPO parameters
    ppo_mini_batch_size: int = field(default=128)
    ppo_micro_batch_size: int = field(default=8)
    ppo_epochs: int = field(default=4)  # Add missing ppo_epochs
    use_dynamic_bsz: bool = field(default=False)

    # Clipping parameters
    clip_ratio_high: float = field(default=0.28)
    clip_ratio_low: float = field(default=0.2)
    grad_clip: float = field(default=1.0)

    # Environment rollout parameters
    rollout_micro_batch_size: int = field(default=1)
    log_prob_micro_batch_size: int = field(default=32)

    # Training schedule
    total_epochs: int = field(default=200)
    save_freq: int = field(default=10)  # default=25
    test_freq: int = field(default=4)

    # FSDP configuration for distributed training
    fsdp_param_offload: bool = field(default=False)
    fsdp_grad_offload: bool = field(default=True)
    fsdp_optimizer_offload: bool = field(default=True)

    # NEW: Data redistribution control
    enable_batch_redistribution: bool = field(
        default=True
    )  # Enable/disable batch redistribution across GPUs


@dataclass
class SimpleVLARLGRPOConfig(GRPOConfig):
    """
    GRPO configuration for SimpleVLA-RL
    """

    pass


@dataclass
class SimpleVLARLEnvironmentConfig(RLEnvironmentConfig):
    """
    Environment configuration for SimpleVLA-RL supporting LIBERO
    """

    env_name: str = field(default="libero")
    task_name: str = field(
        default="libero_10"
    )  # Updated to use task_name instead of task_suite_name
    model_family: str = field(default="openvla")

    # Environment-specific parameters
    unnorm_key: str = field(default="libero_10")
    num_steps_wait: int = field(default=10)

    env_config: Dict = field(
        default_factory=lambda: {
            "suite": "libero_10",  # libero_10, libero_goal, libero_spatial etc.
            "max_episode_steps": 600,
            "obs_modality": ["rgb", "proprio"],
            "camera_names": ["agentview", "robot0_eye_in_hand"],
        }
    )


class SimpleVLARLExp(BaseExp):
    """
    SimpleVLA-RL experiment class implementing GRPO for robotic manipulation

    This experiment integrates:
    - OpenVLA model architecture
    - GRPO algorithm for RL training
    - Support for LIBERO environments
    - Distributed training with FSDP
    """

    model_config: SimpleVLAActorRolloutRefConfig = field(
        default_factory=SimpleVLAActorRolloutRefConfig
    )
    data_config: SimpleVLARLDataConfig = field(default_factory=SimpleVLARLDataConfig)
    trainer_config: SimpleVLARLTrainerConfig = field(
        default_factory=SimpleVLARLTrainerConfig
    )
    tokenizer_config: TokenizerConfig = field(default_factory=TokenizerConfig)

    # RL-specific configurations
    rl_config: SimpleVLARLGRPOConfig = field(default_factory=SimpleVLARLGRPOConfig)
    env_config: SimpleVLARLEnvironmentConfig = field(
        default_factory=SimpleVLARLEnvironmentConfig
    )
    env_wrappers: object = None
    uid_count: int = 0

    def _initialize_rl_train(self):
        """
        Initialize RL training with dataset setup (but NOT environment setup)
        Environment will be created dynamically per batch from RL dataset
        """
        logger.info("Initializing SimpleVLA-RL training...")

        # Initialize base components (creates model, tokenizer, and Trainer)
        # This will also initialize distributed training environment via Trainer
        super()._initialize_train()

        # NOW distributed training is initialized by Trainer
        # Safe to create BufferedRLDataLoader which needs dist.is_initialized()

        logger.info("Creating BufferedRLDataLoader (after distributed init)...")
        self.data_config.rl_dataloader = BufferedRLDataLoader(
            self.data_config.rl_dataset,
            n_sample=self.data_config.n_sample,
            shuffle=getattr(self.data_config, "shuffle", True),
            drop_last=getattr(self.data_config, "drop_last", False),
        )

        # Get RL dataset and dataloader from data_config
        self.rl_dataset = self.data_config.rl_dataset
        self.rl_dataloader = self.data_config.rl_dataloader

        logger.info(
            f"DataLoader created: {self.rl_dataloader.num_batches} batches per epoch, "
            f"batch_size={self.rl_dataset.batch_size}, n_sample={self.data_config.n_sample}"
        )

        # read norm stats
        norm_stats_file = os.path.join(
            self.model_config.model_name_or_path, "norm_stats.json"
        )
        self.rl_dataset.norm_stats = read_normalization_stats(norm_stats_file)
        self.reward_fn = RobRewardManager(num_examine=0, config=self.model_config)
        self.kl_ctrl = FixedKLController(kl_coef=0.0)

        # Override trainer with RL trainer
        trainer_kwargs = {
            "model": self.model,
            "processing_class": self.tokenizer,
            "exp_config": self,
            "rl_config": self.rl_config,
            "env_config": self.env_config,
            "train_dataset": None,  # Will be dynamically created from rollouts
            "data_collator": None,
        }

        self.rl_trainer = DexboticRLTrainer(**trainer_kwargs)
        logger.info("RL trainer initialized")

        if torch.cuda.is_available():
            device = torch.device(f"cuda:{self.local_rank}")
            logger.info(f"Moving model to GPU device: {device}")
            self.model = self.model.to(device=device, dtype=torch.bfloat16)
            logger.info(f"Model successfully moved to {device}")
        else:
            logger.warning("CUDA not available, using CPU")
            self.model = self.model.to(dtype=torch.bfloat16)

    def change_uid(self, filtered_batch_rollouts, n_sample, rank):
        length = len(filtered_batch_rollouts["uid"]) // n_sample
        for i in range(length):
            for j in range(n_sample):
                filtered_batch_rollouts["uid"][i * n_sample + j] = (
                    (rank + 1) * 1000000 + self.uid_count + 1
                )
            self.uid_count += 1
        filtered_batch_rollouts["uid"] = filtered_batch_rollouts["uid"].astype(np.int64)
        return filtered_batch_rollouts

    def train_rl(self):
        """
        Main RL training loop with batch-based environment creation and training
        Each batch performs gradient update immediately after rollout collection

        Supports distributed training with data parallelism:
        - Each GPU collects rollouts from different environments
        - Each GPU processes batch_size * n_sample environments
        - Total data per step = world_size * batch_size * n_sample
        - Gradients are synchronized by DeepSpeed across GPUs
        - Each GPU trains on its own collected data (true data parallelism)
        """
        self._initialize_rl_train()

        # Check if distributed training is enabled
        is_distributed = dist.is_initialized()
        rank = dist.get_rank() if is_distributed else 0
        world_size = dist.get_world_size() if is_distributed else 1

        # Initialize trainer state for tracking global steps
        if not hasattr(self.rl_trainer.state, "global_step"):
            self.rl_trainer.state.global_step = 0

        # Calculate total training steps per GPU
        steps_per_epoch = self.rl_dataloader.num_batches
        total_steps = self.trainer_config.total_epochs * steps_per_epoch

        logger.info(
            f"Starting RL training: {total_steps} steps ({self.trainer_config.total_epochs} epochs × {steps_per_epoch} batches/epoch)"
        )
        if is_distributed:
            logger.info(f"Distributed training on {world_size} GPUs (rank {rank})")

        for epoch in range(self.trainer_config.total_epochs):
            logger.info(
                f"Rank {rank}: RL Training Epoch {epoch + 1}/{self.trainer_config.total_epochs}"
            )

            # Set epoch for distributed sampler (ensures different shuffle each epoch)
            if hasattr(self.rl_dataloader, "set_epoch"):
                self.rl_dataloader.set_epoch(epoch)

            # Reset dataloader iterator for this epoch
            dataloader_iter = iter(self.rl_dataloader)

            # Target number of rollouts per training step
            n_sample = self.data_config.n_sample
            batch_size = self.data_config.batch_size
            env_dup = int(self.data_config.batch_size // 10)

            target_rollouts = self.data_config.target_rollouts_num

            step_in_epoch = 0
            valid_batch = None
            collected_count = 0

            # Accumulate timing and metrics
            total_timing_gen = 0
            total_timing_verify = 0
            total_timing_filter = 0
            accumulated_reward_metrics = {}

            # Keep sampling until we have enough valid rollouts
            logger.info(
                f"Collecting rollouts for step {step_in_epoch + 1} (target: {target_rollouts})..."
            )

            model_saved = False
            while collected_count < target_rollouts:
                try:
                    batch_env_configs = next(dataloader_iter)
                except StopIteration:
                    logger.info(
                        f"Dataloader exhausted at step {step_in_epoch + 1}, collected {collected_count}/{target_rollouts} rollouts"
                    )
                    break

                # Collect rollouts from environments
                with Timer(name="gen", text="{name}: {seconds:.1f} seconds") as timer:
                    batch_rollouts = None
                    merge_count = env_dup
                    while True:
                        batch_rollouts_single = self._collect_batch_rollouts(
                            batch_env_configs, cuda_device=rank
                        )  # Let the method auto-detect CUDA device

                        # Check if rollout collection failed
                        if batch_rollouts_single is None:
                            logger.error(
                                f"Rank {rank}: Failed to collect rollouts, skipping this batch"
                            )
                            break

                        batch_rollouts_single = self._prepare_output_batch(
                            batch_rollouts_single
                        )  # 512-step rollout loop
                        if batch_rollouts is None:
                            batch_rollouts = batch_rollouts_single
                        elif len(batch_rollouts["responses"]) < batch_size * n_sample:
                            batch_rollouts = self.merge_dict(
                                batch_rollouts,
                                batch_rollouts_single,
                                env_dup,
                                merge_count,
                            )
                            merge_count += env_dup
                            # Clean up intermediate data
                            del batch_rollouts_single
                            if (
                                len(batch_rollouts["responses"])
                                >= batch_size * n_sample
                            ):
                                break
                        else:
                            del batch_rollouts_single
                            break
                total_timing_gen += timer.last

                # Skip if no rollouts were collected
                if batch_rollouts is None:
                    logger.warning(
                        f"Rank {rank}: No rollouts collected, skipping verification and filtering"
                    )
                    continue

                # Verify and compute scores
                with Timer(
                    name="verify", text="{name}: {seconds:.1f} seconds"
                ) as timer:
                    with torch.no_grad():
                        (
                            _,
                            reward_metrics,
                            format_metrics,
                            reward_format_metrics,
                        ) = self.reward_fn.verify(batch_rollouts)
                total_timing_verify += timer.last

                # Accumulate metrics (will average later) - only keep essential metrics
                for k, v in reward_metrics.items():
                    if k not in accumulated_reward_metrics:
                        accumulated_reward_metrics[k] = []
                    accumulated_reward_metrics[k].append(v)
                # Skip format_metrics and reward_format_metrics to save memory

                # Apply filtering if enabled
                with Timer(
                    name="acc&trunc_filter", text="{name}: {seconds:.1f} seconds"
                ) as timer:
                    if (
                        self.data_config.filter_accuracy
                        or self.data_config.filter_truncated
                    ):
                        before_count = batch_rollouts["responses"].size(0)
                        logger.info(f"Before filtering: {before_count} rollouts")
                        filtered_batch_rollouts = self.filter(
                            batch_rollouts["acc"].cpu(), batch_rollouts, n_sample
                        )
                    else:
                        # No filtering, use original batch
                        filtered_batch_rollouts = batch_rollouts
                        batch_rollouts = None  # Avoid double reference

                    # Clean up original batch_rollouts if filtering was applied
                    if batch_rollouts is not None:
                        del batch_rollouts

                    # ========== START: Redistribute filtered batch ==========
                    if (
                        is_distributed
                        and self.trainer_config.enable_batch_redistribution
                    ):
                        try:
                            filtered_batch_rollouts = self.change_uid(
                                filtered_batch_rollouts, n_sample, rank
                            )
                            # Ensure all ranks reach this point before proceeding with collective operations
                            torch.cuda.empty_cache()
                            dist.barrier()
                            filtered_batch_rollouts = (
                                redistribute_filtered_batch_circular(
                                    filtered_batch_rollouts, n_sample
                                )
                            )
                            filtered_batch_rollouts["uid"] = (
                                filtered_batch_rollouts["uid"]
                                .astype(str)
                                .astype(object)
                            )
                            print(
                                "\n==========================redistribute_success==========================\n"
                            )
                        except Exception as e:
                            logger.error(
                                f"\n======Rank {rank}: Error during filtered batch redistribution: {e}======\n"
                            )
                            raise
                    elif is_distributed:
                        dist.barrier()
                    # ========== END: Redistribute filtered batch ==========

                    if (
                        self.data_config.filter_accuracy
                        or self.data_config.filter_truncated
                    ):
                        after_count = filtered_batch_rollouts["responses"].size(
                            0
                        )  # Number of trajectories remaining after filtering
                        logger.info(
                            f"After filtering: {after_count} rollouts (kept {100.0 * after_count / max(before_count, 1):.1f}%)"
                        )
                total_timing_filter += timer.last

                # Accumulate valid rollouts
                current_count = filtered_batch_rollouts["responses"].size(0)
                if current_count > 0:
                    if valid_batch is None:
                        valid_batch = filtered_batch_rollouts
                        collected_count = current_count
                        # Clean up
                        del filtered_batch_rollouts
                    else:
                        # Concatenate batches
                        valid_batch = self._concat_batches(
                            valid_batch, filtered_batch_rollouts
                        )
                        # Clean up
                        del filtered_batch_rollouts
                        collected_count = valid_batch["responses"].size(0)

                    logger.info(
                        f"Collected {collected_count}/{target_rollouts} rollouts"
                    )

                # Force cleanup
                torch.cuda.empty_cache()

            # Increment global step
            self.rl_trainer.state.global_step += 1
            global_step = self.rl_trainer.state.global_step
            step_in_epoch += 1

            logger.info(
                f"Rank {rank}: Training on {collected_count} rollouts "
                f"| Global Step: {global_step}/{total_steps} ({100.0 * global_step / total_steps:.1f}%) "
                f"| Epoch: {epoch + 1}/{self.trainer_config.total_epochs}"
            )

            # Monitor GPU memory before training
            if torch.cuda.is_available():
                memory_before = torch.cuda.memory_allocated() / 1024**3  # GB
                logger.info(
                    f"Rank {rank}: GPU memory before training: {memory_before:.2f} GB"
                )

            # Train on collected valid batch
            print(
                f"\n--------------------------rank{rank}_start_training--------------------------\n"
            )
            metrics = self._train_batch_rl(
                valid_batch, epoch, step_in_epoch - 1, global_step, target_rollouts
            )

            # Monitor GPU memory after training
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_diff = memory_after - memory_before
                logger.info(
                    f"Rank {rank}: GPU memory after training: {memory_after:.2f} GB (diff: {memory_diff:+.2f} GB)"
                )

            # Clean up
            del valid_batch
            torch.cuda.empty_cache()

            # Add timing metrics
            metrics["timing/gen"] = total_timing_gen
            metrics["timing/verify"] = total_timing_verify
            metrics["timing/acc&trunc_filter"] = total_timing_filter

            # Batch redistribution metrics removed to avoid distributed communication issues

            # Add accumulated reward metrics (average) - only essential ones
            for k, v_list in accumulated_reward_metrics.items():
                metrics[f"train_verify_score/{k}"] = np.mean(v_list) if v_list else 0.0

            metrics["rollouts/collected"] = collected_count
            metrics["rollouts/target"] = target_rollouts

            torch.cuda.empty_cache()

            # Periodic saving (at end of epoch) - only rank 0
            if (epoch + 1) % self.trainer_config.save_freq == 0 and not model_saved:
                self.rl_trainer._save_checkpoint(self.model, trial={})
                if is_distributed:
                    dist.barrier()

        logger.info(
            f"Rank {rank}: RL training completed! Total steps: {self.rl_trainer.state.global_step}"
        )

    def merge_dict(
        self,
        batch1: Dict,
        batch2: Dict,
        env_dup: int,
        merge_count: int,
    ) -> Dict[str, Any]:
        """
        merge two batches dict key by key with n_sample as a group
        """

        if batch1 is None:
            return batch2

        assert (
            batch1.keys() == batch2.keys()
        ), "batch1 and batch2 must have the same keys and same data type"

        merged = {}

        for k in batch1.keys():
            v1 = batch1[k]
            v2 = batch2[k]

            # ---------- torch.Tensor ----------
            if torch.is_tensor(v1):
                assert torch.is_tensor(v2), f"{k} date type is not identical"

                v1_reshape = v1.reshape(-1, merge_count, *v1.shape[1:])

                v2_reshape = v2.reshape(-1, env_dup, *v2.shape[1:])

                v_reshape = torch.cat([v1_reshape, v2_reshape], dim=1)
                merged[k] = v_reshape.reshape(-1, *v_reshape.shape[2:])

            # ---------- numpy.ndarray ----------
            elif isinstance(v1, np.ndarray):
                assert isinstance(v2, np.ndarray), f"{k} date type is not identical"

                v1_reshape = v1.reshape(-1, merge_count, *v1.shape[1:])
                v2_reshape = v2.reshape(-1, env_dup, *v2.shape[1:])

                v_reshape = np.concatenate([v1_reshape, v2_reshape], axis=1)
                merged[k] = v_reshape.reshape(-1, *v_reshape.shape[2:])

            else:
                raise TypeError(f"{k}: not supported type {type(v1)}")

        return merged

    def _train_batch_rl(
        self,
        batch: Dict[str, Any],
        epoch: int,
        batch_idx: int,
        global_step: int,
        target_rollouts: int = 32,
    ):
        """
        Train the policy on a single batch's rollouts (immediate gradient update)
        This is like a regular training step in supervised learning

        All ranks participate in training (gradients are synchronized)
        Each rank trains on its own data (true data parallelism)
        Metrics are aggregated across ranks for logging

        Args:
            batch: Batch data containing rollouts
            epoch: Current epoch number
            batch_idx: Current batch index within epoch
            global_step: Global training step counter
            target_rollouts: Target number of rollouts to train on 1 gpu

        Returns:
            metrics: Dictionary of training metrics
        """
        # Check if distributed
        is_distributed = dist.is_initialized()
        rank = dist.get_rank() if is_distributed else 0
        world_size = dist.get_world_size() if is_distributed else 1

        logger.info(
            f"Rank {rank}: Training on batch {batch_idx + 1} rollouts (Global Step: {global_step})..."
        )

        # Initialize metrics dictionary
        metrics = defaultdict(list)

        # Compute rewards (use no_grad to prevent memory buildup)
        with Timer(name="reward", text="{name}: {seconds:.1f} seconds") as timer:
            with torch.no_grad():
                reward_tensor_dict, reward_compute_metrics = self.reward_fn(batch)
                batch["token_level_scores"] = reward_tensor_dict["all"].detach()

                # Process reward tensors and clean up immediately
                for k, v in reward_tensor_dict.items():
                    batch[k] = v.detach() if isinstance(v, torch.Tensor) else v

                # Clean up reward_tensor_dict to free memory
                del reward_tensor_dict

                for k, v in reward_compute_metrics.items():
                    metrics["train_reward/" + k] = v
        metrics["timing/reward"] = timer.last

        # compute rewards. apply_kl_penalty if available
        with Timer(name="adv", text="{name}: {seconds:.1f} seconds") as timer:
            batch, kl_metrics = apply_kl_penalty(
                batch,
                kl_ctrl=self.kl_ctrl,
                kl_penalty=self.rl_config.kl_penalty,
                action_token_len=self.model_config.model.action_dim,
                action_chunks_len=self.model_config.model.chunk_size,
            )
            # Detach kl_metrics tensors to prevent memory leaks
            kl_metrics = {
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in kl_metrics.items()
            }
            metrics.update(kl_metrics)

            # compute advantages
            batch = compute_advantage(
                batch,
                self.rl_config.gamma,
                self.rl_config.lam,
                adv_estimator=self.rl_config.adv_estimator,
                config=self,
            )

            # Detach advantages and returns to prevent memory leaks
            if "advantages" in batch:
                batch["advantages"] = batch["advantages"].detach()
            if "returns" in batch:
                batch["returns"] = batch["returns"].detach()
        metrics["timing/adv"] = timer.last

        # All ranks update policy (gradients synchronized by DeepSpeed/DDP)
        with Timer(name="update_actor", text="{name}: {seconds:.1f} seconds") as timer:
            actor_output = self.rl_trainer.update_policy(batch)
            # Ensure actor_output contains only scalar values (no tensors)
            actor_output = {
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in actor_output.items()
            }
            metrics.update(actor_output)
        metrics["timing/update_actor"] = timer.last

        # Compute data metrics
        with Timer(name="data_metrics", text="{name}: {seconds:.1f} seconds") as timer:
            data_metrics = compute_data_metrics(batch=batch, config=self)
            metrics.update(data_metrics)
        metrics["timing/data_metrics"] = timer.last

        # Aggregate training metrics across ranks
        if is_distributed:
            aggregated_metrics = self._aggregate_metrics(metrics)
        else:
            aggregated_metrics = metrics

        # Log aggregated training metrics (only on rank 0)
        if rank == 0:
            # Use trainer's logging mechanism
            should_log = global_step % self.rl_trainer.args.logging_steps == 0

            if should_log:
                logger.info(
                    f"Step {global_step} | Epoch {epoch + 1} | Batch {batch_idx + 1} | "
                    f"Aggregated metrics (across {world_size} GPU{'s' if world_size > 1 else ''}):"
                )
                for key, value in aggregated_metrics.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"  {key}: {value:.6f}")

                # Also log learning rate if available
                if self.rl_trainer.lr_scheduler is not None:
                    current_lr = self.rl_trainer.lr_scheduler.get_last_lr()[0]
                    logger.info(f"  learning_rate: {current_lr:.2e}")
                    aggregated_metrics["learning_rate"] = current_lr

        # Clean up intermediate tensors to free memory
        cleanup_keys = ["token_level_scores", "advantages", "returns"]
        for key in cleanup_keys:
            if key in batch:
                del batch[key]

        # Force garbage collection and clear CUDA cache
        torch.cuda.empty_cache()

        logger.info(
            f"Rank {rank}: Completed training on batch {batch_idx + 1} (Global Step: {global_step})"
        )

        return aggregated_metrics

    def _aggregate_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Aggregate metrics across all ranks using all_reduce

        Args:
            metrics: Dictionary of metric name to value

        Returns:
            aggregated_metrics: Dictionary with averaged metrics across all ranks
        """
        if not dist.is_initialized():
            return metrics

        aggregated = {}
        world_size = dist.get_world_size()

        for key, value in metrics.items():
            # Skip None values
            if value is None:
                continue

            # Convert to tensor for all_reduce
            if not isinstance(value, torch.Tensor):
                value_tensor = torch.tensor(
                    value, dtype=torch.float32, device=torch.cuda.current_device()
                )
            else:
                value_tensor = value.clone().detach().float()
                if not value_tensor.is_cuda:
                    value_tensor = value_tensor.cuda()

            # All-reduce to sum across all ranks
            dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)

            # Average by world size
            aggregated[key] = (value_tensor / world_size).item()

        return aggregated

    def _concat_batches(
        self, batch1: Dict[str, Any], batch2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Concatenate two batch dictionaries

        Args:
            batch1: First batch dictionary
            batch2: Second batch dictionary

        Returns:
            Combined batch dictionary
        """
        combined_batch = {}

        # Determine target CUDA device - prioritize finding a CUDA device
        target_device = None

        # First, try to find a CUDA device from batch1
        for key in batch1.keys():
            if isinstance(batch1[key], torch.Tensor) and batch1[key].numel() > 0:
                if batch1[key].device.type == "cuda":
                    target_device = batch1[key].device
                    break

        # If no CUDA device in batch1, try batch2
        if target_device is None:
            for key in batch2.keys():
                if isinstance(batch2[key], torch.Tensor) and batch2[key].numel() > 0:
                    if batch2[key].device.type == "cuda":
                        target_device = batch2[key].device
                        break

        # If still no CUDA device found, use current rank's CUDA device as fallback
        if target_device is None:
            if dist.is_initialized():
                target_device = torch.device(f"cuda:{dist.get_rank()}")
            elif torch.cuda.is_available():
                target_device = torch.device("cuda:0")

        for key in batch1.keys():
            if key not in batch2:
                # If key only in batch1, keep it as is (move to target device if tensor)
                if isinstance(batch1[key], torch.Tensor) and target_device is not None:
                    combined_batch[key] = batch1[key].to(target_device)
                else:
                    combined_batch[key] = batch1[key]
            elif isinstance(batch1[key], torch.Tensor) and isinstance(
                batch2[key], torch.Tensor
            ):
                # Ensure both tensors are on the target CUDA device before concatenation
                tensor1 = batch1[key]
                tensor2 = batch2[key]

                if target_device is not None:
                    tensor1 = tensor1.to(target_device)
                    tensor2 = tensor2.to(target_device)
                elif tensor1.device != tensor2.device:
                    # Fallback: move tensor2 to tensor1's device
                    tensor2 = tensor2.to(tensor1.device)

                # Concatenate tensors along batch dimension (dim=0)
                combined_batch[key] = torch.cat([tensor1, tensor2], dim=0)
            elif isinstance(batch1[key], np.ndarray) and isinstance(
                batch2[key], np.ndarray
            ):
                # Concatenate numpy arrays
                combined_batch[key] = np.concatenate([batch1[key], batch2[key]], axis=0)
            elif isinstance(batch1[key], list) and isinstance(batch2[key], list):
                # Concatenate lists
                combined_batch[key] = batch1[key] + batch2[key]
            else:
                # For non-tensor, non-list fields (e.g., metadata), keep from batch1
                combined_batch[key] = batch1[key]

        # Add any keys that are only in batch2
        for key in batch2.keys():
            if key not in batch1:
                # Move to target device if tensor
                if isinstance(batch2[key], torch.Tensor) and target_device is not None:
                    combined_batch[key] = batch2[key].to(target_device)
                else:
                    combined_batch[key] = batch2[key]

        return combined_batch

    def filter(self, acc_tensor, batch, n_sample):
        """
        Filter responses based on accuracy and truncation criteria.

        Args:
            acc_tensor: Tensor containing accuracy scores for each trajectory (batch_size,)
            batch: Dict containing batch data with keys like 'responses', 'attention_mask', etc.
            n_sample: Number of samples per environment (from data_config.n_sample)

        Returns:
            Dict: Filtered batch with same structure as input
        """

        batch_size = acc_tensor.size(0)
        num_prompts = batch_size // n_sample

        # First do accuracy filtering if enabled
        if self.data_config.filter_accuracy:
            # Reshape to (num_prompts, n_sample) and compute mean accuracy per prompt
            acc_matrix = acc_tensor.reshape(num_prompts, n_sample)
            acc_per_prompt = torch.mean(acc_matrix, dim=-1)

            counts = Counter(acc_per_prompt.tolist())
            logger.info(
                "Accuracy distribution: "
                + " ".join(f"{k:.2f}:{v}" for k, v in sorted(counts.items()))
            )

            acc_mask = (acc_per_prompt >= self.data_config.accuracy_lower_bound) & (
                acc_per_prompt <= self.data_config.accuracy_upper_bound
            )
        else:
            # If accuracy filtering disabled, keep all samples
            acc_mask = torch.ones(
                num_prompts, dtype=torch.bool, device=acc_tensor.device
            )

        # Then do truncation filtering if enabled
        if self.data_config.filter_truncated:
            attention_mask = batch["attention_mask"]  # (batch_size, seq_len)

            # Calculate actual response lengths from attention mask
            response_lengths = attention_mask.sum(-1)  # (batch_size,)
            response_lengths = response_lengths.reshape(
                num_prompts, n_sample
            )  # (num_prompts, n_sample)

            # Get max possible length from config
            max_len = self.data_config.max_response_length

            # Check if any response in the group hits max length (indicating possible truncation)
            has_truncated = (response_lengths >= max_len).any(dim=-1)

            # Print distribution of truncated vs non-truncated
            truncated_counts = Counter(has_truncated.tolist())
            logger.info(
                f"Truncation distribution: "
                f"Truncated: {truncated_counts.get(True, 0)}, "
                f"Non-truncated: {truncated_counts.get(False, 0)}"
            )
            # Keep only prompts where no response was truncated
            trunc_mask = ~has_truncated
        else:
            # If truncation filtering disabled, keep all samples
            trunc_mask = torch.ones(
                num_prompts, dtype=torch.bool, device=acc_tensor.device
            )

        # Combine both masks
        combined_mask = acc_mask & trunc_mask

        # Expand mask to cover all samples for each prompt
        final_mask = combined_mask.repeat_interleave(n_sample)

        # Apply the mask to the batch - filter all tensor fields
        filtered_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                filtered_batch[key] = value[final_mask]
            elif isinstance(value, np.ndarray):
                filtered_batch[key] = value[final_mask]
            elif isinstance(value, list):
                # For list fields, apply mask
                filtered_batch[key] = [v for i, v in enumerate(value) if final_mask[i]]
            else:
                # Keep non-tensor fields as is (metadata)
                filtered_batch[key] = value

        logger.info(
            f"Filtered batch size: {final_mask.sum().item()} "
            f"(from original size: {batch_size}, kept {100 * final_mask.sum().item() / batch_size:.1f}%)"
        )

        return filtered_batch

    def _prepare_output_batch(self, batch_rollouts: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare the output batch from VLA history"""
        vla_history = batch_rollouts["vla_history"]
        task_records = batch_rollouts["task_records"]
        uid = batch_rollouts["uid"]

        batch = {
            "responses": [],
            "input_ids": [],
            "attention_mask": [],
            "pixel_values": [],
            "log_probs": [],
            "entropy": [],
        }

        batch["uid"] = uid

        key_names = [
            "responses",
            "input_ids",
            "attention_mask",
            "pixel_values",
            "log_probs",
            "entropy",
        ]

        for k in key_names:
            for h in vla_history:
                batch[k].append(h[k])

        for k, v in batch.items():
            if k in ["uid"]:
                continue
            batch[k] = torch.stack(v, dim=1)

        batch["complete"] = torch.tensor(
            [bool(k["complete"]) for k in task_records],
            dtype=torch.bool,
            device=batch["responses"].device,
        )
        batch["finish_step"] = torch.tensor(
            [k["finish_step"] for k in task_records],
            dtype=torch.int64,
            device=batch["responses"].device,
        )

        #
        batch_size, traj_len, chunk_size, action_dim = batch["log_probs"].size()

        batch["log_probs"] = batch["log_probs"].reshape(
            (batch_size, traj_len * chunk_size, action_dim)
        )  # *
        batch["entropy"] = batch["entropy"].reshape(
            (batch_size, traj_len * chunk_size, action_dim)
        )

        mask = self.rl_trainer.generate_traj_mask(
            batch["finish_step"], traj_len * chunk_size
        )  # , self.config.action_token_len
        (
            batch["log_probs"],
            batch["entropy"],
        ) = self.rl_trainer.apply_mask_with_grad_control(
            batch["log_probs"], batch["entropy"], mask
        )

        batch["old_log_probs"] = batch["log_probs"].reshape(
            (batch_size, traj_len * chunk_size * action_dim)
        )
        batch["old_entropy"] = batch["entropy"].reshape(
            (batch_size, traj_len * chunk_size * action_dim)
        )

        # Clean up intermediate log_probs from batch
        del batch["log_probs"]
        del batch["entropy"]

        torch.cuda.empty_cache()
        return batch

    def process_input(self, inputs: List[Dict], task_descriptions: List[str]) -> tuple:
        """
        Process inputs and task descriptions to prepare data for inference_action.
        Reference: oft_exp.py _get_response method

        Args:
            inputs: List of input dictionaries containing images and states
            task_descriptions: List of task description strings

        Returns:
            tuple: (input_ids, image_tensors, inference_args_list)
        """
        batch_size = len(inputs)
        input_ids_list = []
        attention_masks_list = []
        image_tensors_list = []

        for idx in range(batch_size):
            inp = inputs[idx]
            task_desc = task_descriptions[idx]

            # Process images
            images = []
            if "full_image" in inp:
                if isinstance(inp["full_image"], np.ndarray):
                    images.append(Image.fromarray(inp["full_image"]).convert("RGB"))
                else:
                    images.append(inp["full_image"].convert("RGB"))

            if "wrist_image" in inp:
                if isinstance(inp["wrist_image"], np.ndarray):
                    images.append(Image.fromarray(inp["wrist_image"]).convert("RGB"))
                else:
                    images.append(inp["wrist_image"].convert("RGB"))

            # Process images through model
            if len(images) == 1:
                image_tensor = self.model.process_images(images).to(
                    dtype=self.model.dtype
                )
            else:
                image_tensor = (
                    self.model.process_images(images)
                    .to(dtype=self.model.dtype)
                    .unsqueeze(0)
                )

            image_tensors_list.append(image_tensor)

            # Create conversation template
            conv = conversation_lib.conv_templates[
                self.model.config.chat_template
            ].copy()
            text = f"What action should the robot take to {task_desc}?"
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + text)
            conv.append_message(conv.roles[1], " ")
            prompt = conv.get_prompt()

            # Tokenize
            input_ids = (
                tokenizer_image_token(
                    prompt.replace("  ", " "),
                    self.tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                )
                .unsqueeze(0)
                .to(self.model.device)
            )

            input_ids_list.append(input_ids)
            # attention_masks_list.append(input_ids.ne(self.tokenizer.pad_token_id))
            attention_masks_list.append(torch.ones_like(input_ids))

            # Prepare inference args
            inference_args = {"action_norms": self.rl_dataset.norm_stats}

        # Padding to fixed length
        max_prompt_length = self.data_config.max_prompt_length

        # First transpose for pad_sequence
        input_ids = [x.transpose(0, 1) for x in input_ids_list]
        attention_masks = [x.transpose(0, 1) for x in attention_masks_list]

        # Pad to longest sequence in batch
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).squeeze(-1)
        attention_masks = pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        ).squeeze(-1)

        # Get current length
        current_length = input_ids.size(1)

        # Pad or truncate to max_prompt_length
        if current_length < max_prompt_length:
            # Need to pad more
            pad_length = max_prompt_length - current_length
            input_ids = torch.nn.functional.pad(
                input_ids, (0, pad_length), value=self.tokenizer.pad_token_id
            )
            attention_masks = torch.nn.functional.pad(
                attention_masks, (0, pad_length), value=0
            )
        elif current_length > max_prompt_length:
            # Need to truncate
            logger.warning(
                f"Input sequence length ({current_length}) exceeds max_prompt_length ({max_prompt_length}). "
                f"Truncating from the left (keeping most recent tokens)."
            )
            # Truncate from left (keep right side which is more recent)
            input_ids = input_ids[:, -max_prompt_length:]
            attention_masks = attention_masks[:, -max_prompt_length:]

        pixel_values = torch.cat(image_tensors_list, dim=0)

        return input_ids, pixel_values, attention_masks, inference_args

    def _collect_batch_rollouts(
        self, batch_env_configs: Dict[str, Any], cuda_device=None
    ) -> Dict[str, Any]:
        """
        Collect rollouts for a specific batch by:
        1. Receiving batch environment configurations from RL dataloader
        2. Creating environments for this batch
        3. Running rollouts
        4. Cleaning up environments

        Args:
            batch_env_configs: Dictionary containing batch environment configurations
                              from the dataloader (after n_sample interleaving)

        Returns:
            rollout_data: Dictionary containing rollout trajectories and rewards
        """

        # Set model to eval mode for rollout collection
        self.model.eval()

        # Create environments for this batch (first time) or request fresh init state (subsequent times)
        if self.env_wrappers is None:
            self.env_wrappers = self._create_batch_environments(
                batch_env_configs, cuda_device=cuda_device
            )
        else:
            # Request all environments to send fresh init state for new rollout
            logger.info("Requesting fresh init state from all environments...")
            for env in self.env_wrappers:
                env.input_queue.put(None)  # Signal to send init state

            # Collect init data from all environments
            for env in self.env_wrappers:
                init_data = env.output_queue.get(timeout=120)
                if init_data["type"] == "error":
                    raise RuntimeError(f"Environment error: {init_data['message']}")
                assert (
                    init_data["type"] == "init"
                ), f"Expected 'init' but got '{init_data['type']}'"
                env.init_data = init_data  # Update init_data
            logger.info("All environments ready with fresh init state")

        temperature = self.model_config.rollout.temperature
        inputs = []
        task_descriptions = []
        task_records = []

        try:
            for env in self.env_wrappers:
                init_data = env.init_data
                assert init_data["type"] == "init"
                task_descriptions.append(init_data["task_description"])
                inputs.append(self._obs_to_input(init_data["obs"]))
                task_records.append(
                    {
                        "active": init_data["active"],
                        "complete": init_data["complete"],
                        "finish_step": init_data["finish_step"],
                        "task_file_name": init_data["task_file_name"],
                    }
                )

            step = 0
            # Initialize vla_history to record inference inputs/outputs for each environment
            vla_history = []

            while step < 512:  # All environments run for 512 steps
                # Get active environment indices
                active_indices = [i for i, r in enumerate(task_records) if r["active"]]

                with torch.no_grad():
                    # Process inputs and task descriptions for active environments only
                    (
                        input_ids,
                        pixel_values,
                        attention_masks,
                        inference_args,
                    ) = self.process_input(inputs, task_descriptions)

                    # Call generate_action
                    actions, response = self.model.generate_action(
                        input_ids,
                        pixel_values,
                        attention_masks,
                        temperature,
                        inference_args,
                    )

                # Predict actions for each active environment ONLY
                for idx in active_indices:
                    env = self.env_wrappers[idx]
                    # Send action to environment
                    env.input_queue.put(np.array(actions[idx]))

                # Collect results from active environments
                new_inputs = inputs.copy()
                for idx in active_indices:
                    result = self.env_wrappers[idx].output_queue.get(timeout=30)
                    assert (
                        result["type"] == "step"
                    ), f"Expected 'step' but got '{result['type']}'"
                    # Update observation
                    new_inputs[idx] = self._obs_to_input(result["obs"])

                    # Update task records
                    task_records[idx]["active"] = result["active"]
                    task_records[idx]["complete"] = result["complete"]
                    task_records[idx]["finish_step"] = result["finish_step"]

                entropy, log_probs = self.rl_trainer.compute_log_prob(
                    data={
                        "responses": response.unsqueeze(1),
                        "input_ids": input_ids.unsqueeze(1),
                        "attention_mask": attention_masks.unsqueeze(1),
                        "pixel_values": pixel_values.unsqueeze(1),
                    },
                    masked=False,
                )

                # Record inference history for this environment
                vla_history.append(
                    {
                        "responses": response.detach(),  # bs, chunk_size * action_dim
                        "input_ids": input_ids.detach(),
                        "attention_mask": attention_masks.detach(),
                        "pixel_values": pixel_values.detach(),
                        "actions": actions.detach()
                        if isinstance(actions, torch.Tensor)
                        else actions,
                        "log_probs": log_probs.detach(),
                        "entropy": entropy.detach(),
                        "step": step,
                        # 'task_description': task_descriptions,
                        # 'observation': inputs,
                        # 'states': inference_args.get('states', None),
                    }
                )

                logger.info(
                    f"Step {step}: Predicted actions for {len(active_indices)} active environments"
                )

                inputs = new_inputs
                step += self.model_config.model.chunk_size

                if step % 8 == 0:  # Every 1 chunks (8 steps)
                    torch.cuda.empty_cache()

            # Prepare rollout data
            rollout_data = {
                "vla_history": vla_history,
                "task_records": task_records,
                "uid": batch_env_configs["uid"],
            }

            logger.info(
                f"Collected rollouts for batch: {len(self.env_wrappers)} environments, {step} steps"
            )

            # reset all environments
            logger.info("environments reseting...")

            return rollout_data

        except Exception as e:
            # Clean up environments in case of error
            logger.error(f"Error during rollout collection: {e}")
            logger.info("Cleaning up environments after error...")

    def _create_batch_environments(self, batch_env_configs: Dict[str, Any]):
        """
        Create environments for a specific batch based on RL dataset configurations

        This method should be implemented by subclasses to create environment managers
        specific to their environment type (LIBERO...)

        Args:
            batch_env_configs: Dictionary containing batch environment configurations
                              from the dataloader (after n_sample interleaving)

        Returns:
            env_manager: Environment manager instance (e.g., EnvBatchManager)

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError(
            "Subclasses must implement _create_batch_environments() to create "
            "environment-specific batch managers (e.g., using EnvBatchManager)"
        )

    def _obs_to_input(self, obs):
        """Convert observation to model input format for LIBERO"""
        state = np.concatenate(
            [
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            ]
        )

        if self.data_config.num_images > 1:
            return {
                "full_image": get_libero_image(obs, 224),
                "wrist_image": get_libero_wrist_image(obs, 224),
                "state": state,
            }
        else:
            return {"full_image": get_libero_image(obs, 224), "state": state}

    def train(self):
        """
        Override base train method to use RL training
        """
        self.train_rl()


if __name__ == "__main__":
    # Example usage for LIBERO
    exp = SimpleVLARLExp()

    # Configure for LIBERO
    exp.env_config.env_name = "libero"
    exp.env_config.task_name = "libero_10"
    exp.data_config.env_type = "libero"
    exp.data_config.task_name = "libero_10"
    exp.data_config.batch_size = 8
    exp.data_config.n_sample = 4

    # Configure model path (should be set to actual SFT model path)
    exp.model_config.model_name_or_path = "/path/to/sft/model"

    # Configure output directory
    exp.trainer_config.output_dir = "/path/to/output"

    # Start training
    exp.train()
