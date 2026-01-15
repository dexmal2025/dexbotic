import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Optional, cast

import torch
from flask import jsonify, request
from loguru import logger
from PIL import Image
from transformers import AutoTokenizer

from dexbotic.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from dexbotic.exp.base_exp import (
    ActionConfig,
    BaseExp,
    ComputeNormActionConfig,
    DataConfig,
)
from dexbotic.exp.base_exp import InferenceConfig as BaseInferenceConfig
from dexbotic.exp.base_exp import ModelConfig, OptimizerConfig, TrainerConfig
from dexbotic.model.oft.oft_discrete_arch import (
    OFTDiscreteConfig,
    OFTDiscreteForCausalLM,
    OFTDiscreteModel,
)
from dexbotic.tokenization import conversation as conversation_lib
from dexbotic.tokenization.tokenization import tokenizer_image_token


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        choices=["train", "inference", "compute_norm_stats"],
    )
    args, unknown = parser.parse_known_args()
    return args


@dataclass
class OFTDiscreteOptimizerConfig(OptimizerConfig):
    base_lr: float = field(default=2e-5)


@dataclass
class OFTDiscreteTrainerConfig(TrainerConfig):
    num_train_epochs: int = field(default=5)
    save_steps: int = field(default=20000)
    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=2)


@dataclass
class OFTDiscreteActionConfig(ActionConfig):
    pass


@dataclass
class OFTDiscreteDataConfig(DataConfig):
    action_config: ActionConfig = field(default_factory=OFTDiscreteActionConfig)


@dataclass
class OFTDiscreteModelConfig(ModelConfig):
    """
    OFT-Discrete Model Configuration Class - Inherits from base model configuration

    Configuration details:
    - action_model_type: Action model type, choices: 'DiT', 'Linear'
    - action_dim: Action dimension, typically 7 (position + rotation + gripper)
    - chunk_size: Action chunk size
    - freeze_action_head: Whether to freeze the action head module
    - use_proprio: Whether to use proprioception
    - proprio_dim: Proprioception dimension
    """

    action_model_type: str = field(default="DiT")
    action_dim: int = field(default=7)
    chunk_size: int = field(default=16)
    freeze_action_head: bool = field(default=False)
    use_proprio: bool = field(default=False)
    proprio_dim: int = field(default=7)

    def build_model(self) -> OFTDiscreteForCausalLM:
        if self.from_llm:
            model_config_args = {
                "llm_config": self.model_name_or_path,
                "chat_template": self.chat_template,
                "mm_projector_type": self.mm_projector_type,
                "mm_vision_tower": self.mm_vision_tower,
                "action_model_type": self.action_model_type,
                "action_dim": self.action_dim,
                "chunk_size": self.chunk_size,
                "init_llm_weights": True,
                "use_proprio": self.use_proprio,
                "proprio_dim": self.proprio_dim,
            }
            model_config = OFTDiscreteConfig(**model_config_args)
            model = OFTDiscreteForCausalLM(model_config)
        else:
            model_config_args = {
                "model_name_or_path": self.model_name_or_path,
                "mm_projector_type": self.mm_projector_type,
                "mm_vision_tower": self.mm_vision_tower,
                "action_model_type": self.action_model_type,
                "action_dim": self.action_dim,
                "chunk_size": self.chunk_size,
                "use_proprio": self.use_proprio,
                "proprio_dim": self.proprio_dim,
            }
            model = OFTDiscreteForCausalLM.from_pretrained(self.model_name_or_path)
            model.model.initialize_model(model_config_args)

        self._freeze_model(model)

        return model

    def _freeze_model(self, model: OFTDiscreteForCausalLM):
        model.model = cast(OFTDiscreteModel, model.model)

        # set requires_grad to True for all parameters
        for param in model.model.parameters():
            param.requires_grad = True

        if self.freeze_llm:
            for param in model.model.backbone.parameters():
                param.requires_grad = False
        if self.freeze_mm_projector:
            for param in model.model.mm_projector_module.parameters():
                param.requires_grad = False
        if self.freeze_mm_vision:
            for param in model.model.mm_vision_module.parameters():
                param.requires_grad = False
        if self.freeze_action_head:
            for param in model.model.action_head_module.parameters():
                param.requires_grad = False


@dataclass
class InferenceConfig(BaseInferenceConfig):
    def process_frame(self) -> None:
        results = self._get_response(
            text=request.form.get("text"),
            images=request.files.getlist("image"),
            states=request.form.get("states", None),
        )
        return jsonify({"response": results})

    def _load_model(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model from {self.model_name_or_path}")
        logger.info(f"Using device: {self.device}")
        model = OFTDiscreteForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map={"": "cuda:0"},
        ).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model.config
        logger.info("Model loaded successfully")

    def _get_response(
        self, text: str, images: list[str], states: Optional[str] = None
    ) -> str:
        t0 = time.monotonic()
        if len(images) == 1:
            images = [Image.open(images[0]).convert("RGB")]
            image_tensor = self.model.process_images(images).to(dtype=self.model.dtype)
        else:
            images = [Image.open(image).convert("RGB") for image in images]
            image_tensor = (
                self.model.process_images(images)
                .to(dtype=self.model.dtype)
                .unsqueeze(0)
            )

        self._save_image(images, text)

        conv = conversation_lib.conv_templates[self.model_config.chat_template].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + text)
        conv.append_message(conv.roles[1], " ")
        prompt = conv.get_prompt()
        prompt = prompt.replace("  ", " ")
        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.model.device)
        )
        logger.debug(f"input_ids: {input_ids}")
        inference_args = {
            "cfg_scale": 1.5,
            "num_ddim_steps": 10,
            "action_norms": self.norm_stats,
        }

        if states is not None:
            states = json.loads(states)
            states = torch.tensor(
                states, dtype=self.model.dtype, device=self.model.device
            ).reshape(1, -1)
            inference_args["states"] = states

        outputs = self.model.inference_action(input_ids, image_tensor, inference_args)
        for i, out in enumerate(outputs):
            outputs[i][-1] = 1.0 if out[-1] < 0.5 else -1.0
        logger.info(f"prompt: <start>{prompt}<end>\naction: {outputs}")
        logger.info(f"Processing time: {time.monotonic() - t0}")
        return outputs


@dataclass
class OFTDiscreteExp(BaseExp):
    model_config: OFTDiscreteModelConfig = field(default_factory=OFTDiscreteModelConfig)
    optimizer_config: OFTDiscreteOptimizerConfig = field(
        default_factory=OFTDiscreteOptimizerConfig
    )
    trainer_config: OFTDiscreteTrainerConfig = field(
        default_factory=OFTDiscreteTrainerConfig
    )
    data_config: OFTDiscreteDataConfig = field(default_factory=OFTDiscreteDataConfig)
    inference_config: InferenceConfig = field(default_factory=InferenceConfig)

    def inference(self) -> None:
        self.inference_config.run()

    def compute_norm_stats(self) -> None:
        self.data_config.action_config = ComputeNormActionConfig()
        self.data_config.action_config.compute_norm_stats(self.data_config.dataset_name)


if __name__ == "__main__":
    args = parse_args()
    exp = OFTDiscreteExp()
    if args.task == "train":
        exp.train()
    elif args.task == "inference":
        exp.inference()
    elif args.task == "compute_norm_stats":
        exp.compute_norm_stats()
