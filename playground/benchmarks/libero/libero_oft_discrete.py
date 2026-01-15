"""
Libero OFT Discrete Experiment Configuration
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime

from dexbotic.exp.oft_discrete_exp import (
    InferenceConfig,
    OFTDiscreteActionConfig,
    OFTDiscreteDataConfig,
    OFTDiscreteExp,
    OFTDiscreteModelConfig,
    OFTDiscreteTrainerConfig,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        choices=["train", "inference", "inference_single"],
    )
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    args, unknown = parser.parse_known_args()
    return args


@dataclass
class LiberoOFTDiscreteTrainerConfig(OFTDiscreteTrainerConfig):
    output_dir: str = field(
        default=f'./user_checkpoints/dexbotic/libero_oft_discrete/discrete-{datetime.now().strftime("%m%d%H%M")}'
    )
    wandb_project: str = field(default="dexbotic_libero_oft_discrete")
    num_train_epochs: int = field(default=25)
    save_strategy: str = field(default="epoch")
    per_device_train_batch_size: int = field(
        default=16
    )  # Smaller batch size for discrete training
    gradient_accumulation_steps: int = field(default=1)
    save_total_limit: int = field(default=3)


@dataclass
class LiberoOFTDiscreteActionConfig(OFTDiscreteActionConfig):
    from dexbotic.data.dataset.transform.common import Pipeline

    trajectory_length: int = field(default=8)

    def build_action_process_func(self) -> Pipeline:
        action_process_ppl = super().build_action_process_func()
        del action_process_ppl.transforms[-2]  # Remove ReplaceAnswer transform
        return action_process_ppl


@dataclass
class LiberoOFTDiscreteDataConfig(OFTDiscreteDataConfig):
    action_config: LiberoOFTDiscreteActionConfig = field(
        default_factory=LiberoOFTDiscreteActionConfig
    )
    dataset_name: str = field(default="libero_goal")
    auto_norm_method: str = field(default="minmax")
    data_keys: list[str] = field(
        default_factory=lambda: [
            "input_ids",
            "labels",
            "action",
            "image",
        ]
    )


@dataclass
class LiberoOFTDiscreteModelConfig(OFTDiscreteModelConfig):
    """
    Libero OFT Discrete Model Configuration
    Uses discrete action model type for text-based action generation
    """

    # You should put the pre-trained model path here
    model_name_or_path: str = field(default="./checkpoints/Dexbotic-Base")

    # Discrete action configuration
    action_model_type: str = field(default="Discrete")  # Use discrete action head
    action_dim: int = field(default=7)  # Standard robot action dimension
    chunk_size: int = field(default=8)  # Action chunk size
    num_bins: int = field(
        default=256
    )  # Number of discrete bins for action quantization

    # Proprioception configuration (typically not used in Libero)
    use_proprio: bool = field(default=False)
    proprio_dim: int = field(default=7)

    # Action head training configuration
    freeze_action_head: bool = field(default=False)


@dataclass
class LiberoOFTDiscreteInferenceConfig(InferenceConfig):
    # You should put the inference model path here
    model_name_or_path: str = field(default="./checkpoints/libero/libero_oft_discrete")
    port: int = field(default=7892)  # Different port from other experiments


@dataclass
class LiberoOFTDiscreteExp(OFTDiscreteExp):
    model_config: LiberoOFTDiscreteModelConfig = field(
        default_factory=LiberoOFTDiscreteModelConfig
    )
    trainer_config: LiberoOFTDiscreteTrainerConfig = field(
        default_factory=LiberoOFTDiscreteTrainerConfig
    )
    data_config: LiberoOFTDiscreteDataConfig = field(
        default_factory=LiberoOFTDiscreteDataConfig
    )
    inference_config: LiberoOFTDiscreteInferenceConfig = field(
        default_factory=LiberoOFTDiscreteInferenceConfig
    )

    def _initialize_train(self):
        super()._initialize_train()

    def inference_single(self, image_path: str, prompt: str):
        """
        Single inference for discrete OFT model.
        The discrete action head will generate text tokens representing discretized actions.
        """
        self.inference_config._initialize_inference()
        actions = self.inference_config._get_response(prompt, [image_path])
        return actions


if __name__ == "__main__":
    args = parse_args()
    exp = LiberoOFTDiscreteExp()
    if args.task == "train":
        exp.train()
    elif args.task == "inference":
        exp.inference()
    elif args.task == "inference_single":
        exp.inference_single(args.image_path, args.prompt)
