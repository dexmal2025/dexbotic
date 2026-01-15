#!/usr/bin/env python3
# Copyright 2026 DexBotic Team

import argparse
import multiprocessing
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict

# Set multiprocessing method to “spawn”
# This is critical for DeepSpeed + CUDA + multiprocessing compatibility
try:
    multiprocessing.set_start_method("spawn", force=True)
    print("✓ Set multiprocessing start method to 'spawn' for CUDA compatibility")
except RuntimeError as e:
    current_method = multiprocessing.get_start_method()
    print(f"Failed to set multiprocessing start method to 'spawn': {e}")
    print(f"Multiprocessing method already set to '{current_method}'")

# Check transformers version
try:
    import transformers

    required_version = "4.51.0"
    current_version = transformers.__version__
    if current_version != required_version:
        raise ValueError(
            f"transformers version {required_version} is required, but found {current_version}"
        )
    print(f"✓ transformers version {current_version} is compatible")
except ImportError:
    raise ImportError("transformers library is not installed")
except ValueError as e:
    print(f"{e}")
    raise

import sys

if "/app/libero" not in sys.path:
    sys.path.insert(0, "/app/libero")

os.environ["MUJOCO_GL"] = "egl"

from loguru import logger

from dexbotic.data.dataset.dex_rl_dataset import DexRLDataset, FakeDataset
from dexbotic.exp.simplevla_rl_exp import (
    RolloutSubConfig,
    SimpleVLAActorRolloutRefConfig,
    SimpleVLARLDataConfig,
    SimpleVLARLEnvironmentConfig,
    SimpleVLARLExp,
    SimpleVLARLGRPOConfig,
    SimpleVLARLTrainerConfig,
)
from dexbotic.sim_envs import EnvBatchManager


def parse_args():
    parser = argparse.ArgumentParser(
        description="LIBERO SimpleVLA-RL Post-Training (GRPO)",
        epilog="Note: This script is for RL post-training only. "
        "For model evaluation/inference, please use playground/benchmarks/libero/libero_oft_discrete.py",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        choices=["train"],
        help="Task to run (only 'train' is supported for RL post-training)",
    )
    parser.add_argument(
        "--sft_model_path",
        type=str,
        # required=True,
        default="/path/to/sft-checkpoint/",
        help="Path to the SFT (Supervised Fine-Tuned) model checkpoint",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="libero_10",
        choices=[
            "libero_10",
            "libero_spatial",
            "libero_object",
            "libero_goal",
        ],
        help="LIBERO dataset/task suite to use",
    )

    args, unknown = parser.parse_known_args()
    return args


@dataclass
class LiberoSimpleVLAActorRolloutRefConfig(SimpleVLAActorRolloutRefConfig):
    """
    Libero OFT Discrete Model Configuration
    Uses discrete action model type for text-based action generation

    This config provides property delegation to maintain backward compatibility
    with code that expects direct access to model attributes.
    """

    def __post_init__(self):
        """Override default values for nested configurations"""
        super().__post_init__() if hasattr(super(), "__post_init__") else None

        # Model configuration
        self.model.model_name_or_path = "give path in parse_args()"
        self.model.action_dim = 7
        self.model.chunk_size = 8

        # Actor configuration
        self.actor.optim.lr = 5e-6
        self.actor.optim.warmup_style = "constant"
        self.actor.ppo_mini_batch_size = 128
        self.actor.ppo_micro_batch_size = 8
        self.actor.use_dynamic_bsz = False
        self.actor.fsdp_config.param_offload = False
        self.actor.fsdp_config.grad_offload = True
        self.actor.fsdp_config.optimizer_offload = True
        self.actor.grad_clip = 1.0
        self.actor.clip_ratio_high = 0.28
        self.actor.clip_ratio_low = 0.2
        self.actor.num_images_in_input = 1
        self.actor.traj_mini_batch_size = 4
        self.actor.entropy_coeff = 0.0

        # Rollout configuration
        self.rollout.num_images_in_input = 1
        self.rollout.use_proprio = False
        self.rollout.temperature = 1.6
        self.rollout.micro_batch_size = 1
        self.rollout.model_family = "openvla"
        self.rollout.num_steps_wait = 10
        self.rollout.max_prompt_length = 512
        self.rollout.log_prob_micro_batch_size = 32

        # Reference model configuration
        self.ref.log_prob_micro_batch_size = 32
        self.ref.fsdp_config.param_offload = True


@dataclass
class LiberoSimpleVLARLDataConfig(SimpleVLARLDataConfig):
    """
    Data configuration for LIBERO SimpleVLA-RL
    """

    # LIBERO-specific dataset settings
    env_type: str = field(default="libero")  # Set environment type
    task_name: str = field(default="libero_10")  # LIBERO task name
    num_trials_per_task: int = field(default=50)

    # RL dataset parameters (optimized for multi-GPU training)
    batch_size: int = field(default=10)
    n_sample: int = field(default=8)  # Number of samples for interleave duplication
    target_rollouts_num: int = field(default=32)  # Target number of rollouts per gpu

    # Training/validation split
    train_val: str = field(default="train")  # "train" or "valid"

    # Data filtering for LIBERO
    filter_accuracy: bool = field(default=True)
    accuracy_lower_bound: float = field(default=0.1)
    accuracy_upper_bound: float = field(default=0.9)
    oversample_factor: int = field(default=1)

    # Batch configuration
    train_batch_size: int = field(default=2)
    val_batch_size: int = field(default=496)

    # Sequence lengths optimized for LIBERO tasks
    max_prompt_length: int = field(default=256)
    max_response_length: int = field(default=128)

    # Vision configuration for LIBERO
    num_images: int = field(default=1)  # Single camera view
    use_proprio: bool = field(default=False)  # LIBERO typically uses vision-only

    # LIBERO-specific data keys
    data_keys: list[str] = field(
        default_factory=lambda: [
            "input_ids",
            "labels",
            "action",
            "image",
            "attention_mask",
        ]
    )

    def build_data(self, tokenizer, chat_template, image_processor):
        """
        Override build_data to create RL dataset instead of supervised dataset

        For RL training:
        - Creates DexRLDataset with environment configurations
        - DEFERS BufferedRLDataLoader creation until after distributed training is initialized
        - Stores dataset as self.rl_dataset
        - Returns None, None since RL doesn't use traditional dataset/collator
        """

        logger.info("Setting up LIBERO RL dataset...")

        # Create RL dataset
        self.rl_dataset = DexRLDataset(
            env_type=self.env_type,
            task_name=self.task_name,
            batch_size=self.batch_size,
            num_trials_per_task=self.num_trials_per_task,
            train_val=self.train_val,
            seed=getattr(self, "seed", 42),
        )

        logger.info(
            f"RL dataset initialized with {len(self.rl_dataset)} base environment configurations"
        )
        self.rl_dataloader = None

        ret_dataset = FakeDataset()

        # RL training doesn't use traditional dataset and data_collator
        # They will be created dynamically from rollouts
        return ret_dataset, None


@dataclass
class LiberoSimpleVLARLTrainerConfig(SimpleVLARLTrainerConfig):
    """
    Trainer configuration optimized for LIBERO tasks
    """

    # Learning rate schedule
    actor_lr: float = field(default=5e-6)
    warmup_style: str = field(default="constant")

    # PPO parameters tuned for LIBERO
    ppo_mini_batch_size: int = field(default=128)
    ppo_micro_batch_size: int = field(default=8)
    use_dynamic_bsz: bool = field(default=False)

    # Clipping parameters from SimpleVLA-RL paper
    clip_ratio_high: float = field(default=0.28)
    clip_ratio_low: float = field(default=0.2)
    grad_clip: float = field(default=1.0)

    # Temperature for action sampling
    temperature: float = field(default=1.6)

    # Training schedule
    total_epochs: int = field(default=200)
    save_freq: int = field(default=10)
    test_freq: int = field(default=4)
    logging_steps: int = field(default=1)

    # FSDP configuration for distributed training
    fsdp_param_offload: bool = field(default=False)
    fsdp_grad_offload: bool = field(default=True)
    fsdp_optimizer_offload: bool = field(default=True)

    # Output directory
    output_dir: str = field(
        default=f"./user_checkpoints/dexbotic/libero_simplevla_rl/{datetime.now().strftime('%m%d-%H%M')}"
    )


@dataclass
class LiberoSimpleVLARLGRPOConfig(SimpleVLARLGRPOConfig):
    """
    GRPO configuration for SimpleVLA-RL
    """

    # Advantage estimation parameters
    verifier_gamma: float = field(default=1.0)
    reward_model_gamma: float = field(default=1.0)

    gamma: float = field(default=1.0)
    lam: float = field(default=1.0)
    adv_estimator: str = field(default="grpo")
    adv_params: Dict = field(
        default_factory=lambda: {"verifier_gamma": 1.0, "reward_model_gamma": 1.0}
    )
    kl_penalty: str = field(default="kl")  # how to estimate kl divergence
    kl_ctrl: Dict = field(default_factory=lambda: {"type": "fixed", "kl_coef": 0.00})


@dataclass
class LiberoSimpleVLARLEnvironmentConfig(SimpleVLARLEnvironmentConfig):
    """
    Environment configuration for LIBERO simulation
    """

    env_name: str = field(default="libero")
    task_name: str = field(
        default="libero_10"
    )  # Updated to use task_name instead of task_suite_name
    model_family: str = field(default="openvla")

    # LIBERO-specific parameters
    unnorm_key: str = field(default="libero_10")
    num_steps_wait: int = field(default=10)

    # LIBERO environment configuration
    env_config: Dict = field(
        default_factory=lambda: {
            "suite": "libero_10",
            "obs_modality": ["rgb"],  # Vision-only for LIBERO
            "camera_names": ["agentview", "robot0_eye_in_hand"],
            "image_size": (224, 224),
            "control_freq": 20,
            "horizon": 600,
            "reward_shaping": True,
            "model_family": "openvla",
        }
    )


@dataclass
class LiberoRolloutSubConfig(RolloutSubConfig):
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
class LiberoSimpleVLARLExp(SimpleVLARLExp):
    """
    LIBERO-specific SimpleVLA-RL experiment implementation

    This class extends the base SimpleVLA-RL experiment with LIBERO-specific:
    - Environment setup and integration
    - Task loading and management
    - Evaluation metrics
    - Data collection and preprocessing
    """

    model_config: LiberoSimpleVLAActorRolloutRefConfig = field(
        default_factory=LiberoSimpleVLAActorRolloutRefConfig
    )
    data_config: LiberoSimpleVLARLDataConfig = field(
        default_factory=LiberoSimpleVLARLDataConfig
    )
    trainer_config: LiberoSimpleVLARLTrainerConfig = field(
        default_factory=LiberoSimpleVLARLTrainerConfig
    )
    rl_config: LiberoSimpleVLARLGRPOConfig = field(
        default_factory=LiberoSimpleVLARLGRPOConfig
    )
    env_config: LiberoSimpleVLARLEnvironmentConfig = field(
        default_factory=LiberoSimpleVLARLEnvironmentConfig
    )
    rollout_config: LiberoRolloutSubConfig = field(
        default_factory=LiberoRolloutSubConfig
    )

    def init(self):
        """Initialize after dataclass initialization."""
        super().__post_init__() if hasattr(super(), "__post_init__") else None

        # Create environment batch manager (will be reused across batches)
        from easydict import EasyDict

        config = EasyDict(self.env_config.env_config)
        self.env_batch_manager = EnvBatchManager(
            env_type="libero", task_suite_name=self.env_config.task_name, config=config
        )

    def _create_batch_environments(
        self, batch_env_configs: Dict[str, Any], cuda_device=None
    ):
        """
        Create LIBERO environments for a specific batch using EnvBatchManager

        Args:
            batch_env_configs: Dictionary containing batch environment configurations
                              from the dataloader (after n_sample interleaving)

        Returns:
            env_manager: EnvBatchManager instance with created environments
        """
        num_envs = batch_env_configs.get(
            "interleaved_batch_size", batch_env_configs.get("original_batch_size", 0)
        )
        logger.info(f"Creating {num_envs} LIBERO environments for batch")

        # Get current CUDA device for this process (distributed training aware)
        cuda_device = None
        try:
            import torch

            if torch.cuda.is_available():
                # In distributed training, each process should use its local rank
                if torch.distributed.is_initialized():
                    cuda_device = (
                        torch.distributed.get_rank() % torch.cuda.device_count()
                    )
                else:
                    cuda_device = torch.cuda.current_device()
                logger.info(
                    f"Process rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}: Using CUDA device {cuda_device} for environment workers"
                )
        except Exception as e:
            logger.warning(f"Could not determine CUDA device: {e}")

        # Create batch using the manager
        self.env_batch_manager.create_batch(batch_env_configs, cuda_device=cuda_device)

        logger.info(f"Created {num_envs} LIBERO environments")
        return self.env_batch_manager.env_wrappers


def main():
    """
    Main function for LIBERO SimpleVLA-RL post-training

    This script is designed for RL post-training (GRPO) only.
    For model evaluation or inference, please use:
        playground/benchmarks/libero/libero_oft_discrete.py
    """
    args = parse_args()

    # Create experiment configuration
    exp = LiberoSimpleVLARLExp()

    # Configure paths and settings from command line arguments
    exp.model_config.model_name_or_path = args.sft_model_path
    exp.data_config.task_name = args.dataset_name
    exp.data_config.dataset_name = args.dataset_name
    exp.env_config.task_name = args.dataset_name
    exp.env_config.unnorm_key = args.dataset_name
    exp.env_config.env_config["suite"] = args.dataset_name
    exp.rollout_config.unnorm_key = args.dataset_name
    exp.rollout_config.task_suite_name = args.dataset_name
    exp.init()

    # Log configuration
    logger.info("Starting LIBERO SimpleVLA-RL Post-Training (GRPO)")
    logger.info(f"Task suite: {args.dataset_name}")
    logger.info(f"SFT model: {args.sft_model_path}")

    # Run RL training
    if args.task == "train":
        exp.train()
    else:
        logger.error(f"Unsupported task: {args.task}")
        logger.info("=" * 80)
        logger.info("This script is for RL post-training only.")
        logger.info("For model evaluation or inference, please use:")
        logger.info("  playground/benchmarks/libero/libero_oft_discrete.py")
        logger.info("=" * 80)
        raise ValueError(f"Unsupported task: {args.task}")


if __name__ == "__main__":
    main()
