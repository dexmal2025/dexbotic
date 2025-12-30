from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import (AutoConfig, AutoModel, PretrainedConfig,
                          PreTrainedModel, GenerationMixin)
from transformers.modeling_outputs import ModelOutput

from dexbotic.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from dexbotic.model.modules.mm_projector.builder import build_vision_projector
from dexbotic.model.modules.mm_vision.builder import build_vision_tower


class DexboticConfig(PretrainedConfig):
    model_type = "dexbotic"
    llm_config: str | PretrainedConfig
    mm_projector_type: Optional[str] = 'mlp2x_gelu'
    mm_vision_tower: Optional[str] = None
    chat_template: Optional[str] = 'dexbotic'
    init_llm_weights: Optional[bool] = False


@dataclass
class CausalLMOutputDexbotic(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    text_loss: Optional[torch.FloatTensor] = None
    action_loss: Optional[torch.FloatTensor] = None


class DexboticPretrainedModel(PreTrainedModel):
    config: DexboticConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"

    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_flex_attn = True
    _supports_attention_backend = True


class DexboticVLMModel(DexboticPretrainedModel):
    def __init__(self, config: DexboticConfig):
        super().__init__(config)
        if config.init_llm_weights:
            self.llm = AutoModel.from_pretrained(config.llm_config)
            self.config.init_llm_weights = False
        else:
            if isinstance(config.llm_config, str):
                llm_config = AutoConfig.from_pretrained(config.llm_config)
            elif isinstance(config.llm_config, PretrainedConfig):
                llm_config = config.llm_config
            self.llm = AutoModel.from_config(llm_config)
        self._merge_llm()
        if getattr(config, 'mm_vision_tower', None) is not None:
            self.mm_vision_tower = self._build_mm_vision_module(config.mm_vision_tower)
        else:
            self.mm_vision_tower = self._build_mm_vision_module(config)
        self.mm_projector = self._build_mm_projector_module(config)

        self.post_init()

    def initialize_model(self, extra_config: dict):
        for key, value in extra_config.items():
            setattr(self.config, key, value)
        if getattr(self.config, 'mm_vision_tower', None) is not None:
            self.mm_vision_tower = self._build_mm_vision_module(self.config.mm_vision_tower)
        else:
            self.mm_vision_tower = self._build_mm_vision_module(self.config)
        self.mm_projector = self._build_mm_projector_module(self.config)

    def _merge_llm(self):
        # merge llm config with self.config, only add missing keys
        llm_config_dict = {k: v for k, v in self.llm.config.__dict__.items()
                           if not k.startswith('_') and not hasattr(self.config, k)}
        for key, value in llm_config_dict.items():
            setattr(self.config, key, value)
        self.llm.resize_token_embeddings(self.config.vocab_size)

    def _build_mm_projector_module(self, config) -> nn.Module:
        if getattr(self, 'mm_projector', None) is not None:
            return self.mm_projector
        self.mm_projector = build_vision_projector(config)

        return self.mm_projector

    def _build_mm_vision_module(self, config) -> nn.Module:
        if getattr(self, 'mm_vision_tower', None) is not None:
            return self.mm_vision_tower
        if getattr(config, 'vision_config', None) is not None and getattr(config, 'processor_config', None) is not None:
            # FIXME: processor should be moved to top level config
            self.mm_vision_tower = build_vision_tower(config.vision_config, processor_config=config.processor_config, select_layer=None)
        else: 
            self.mm_vision_tower = build_vision_tower(config)
        self.config.mm_hidden_size = self.mm_vision_tower.hidden_size

        return self.mm_vision_tower

    def _load_pretrain_projector(self, pretrain_mm_mlp_adapter) -> None:
        if pretrain_mm_mlp_adapter is not None:
            print(
                f"=> loading pretrain_mm_mlp_adapter from {pretrain_mm_mlp_adapter} ...")
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k,
                        v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(
                get_w(
                    mm_projector_weights,
                    'mm_projector'),
                strict=True)

    @property
    def mm_projector_module(self) -> nn.Module:
        return self.mm_projector

    @property
    def mm_projector_prefix(self) -> str:
        return "mm_projector"

    @property
    def mm_vision_module(self) -> nn.Module:
        return self.mm_vision_tower

    @property
    def mm_vision_prefix(self) -> str:
        return "mm_vision"

    @property
    def backbone(self) -> PreTrainedModel:
        return self.llm

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.llm = decoder

    def get_decoder(self):
        return self.llm

    def _extract_vision_features(self, images: torch.Tensor) -> torch.Tensor:
        def encode_image(image: torch.Tensor) -> torch.Tensor:
            image_features = self.mm_vision_module(image)
            image_features = self.mm_projector_module(image_features)
            return image_features

        if images.ndim == 5:
            # [B n_image, C, H, W] -> [B*n_image, C, H, W]
            concat_images = torch.cat([image for image in images], dim=0)
            concat_image_features = encode_image(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(
                concat_image_features,
                split_sizes,
                dim=0)  # {[n_image n_token C] * B}
            # {[n_image*n_token C] * B}
            image_features = [x.flatten(0, 1) for x in image_features]
            image_features = torch.stack(
                image_features, dim=0)  # [B, n_image*n_token, C]
        else:
            image_features = encode_image(images)

        image_features = image_features.to(self.device)
        return image_features

    def _prepare_inputs_labels_for_multimodal(self,
                                              input_ids: Optional[torch.Tensor],
                                              position_ids: Optional[torch.Tensor],
                                              attention_mask: Optional[torch.Tensor],
                                              past_key_values: Optional[torch.Tensor],
                                              labels: Optional[torch.Tensor],
                                              cache_position: Optional[torch.Tensor],
                                              images: Optional[torch.Tensor]) -> tuple:
        # in the decode stage
        if input_ids.shape[1] == 1:
            return self._prepare_inputs_labels_for_multimodal_decode(
                input_ids, position_ids, attention_mask, past_key_values, labels, cache_position, images)

        # in the prefill stage
        vision = self.mm_vision_module

        if vision is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, cache_position

        image_features = self._extract_vision_features(images)

        _labels, _position_ids, _attention_mask = labels, position_ids, attention_mask

        # Let's just add dummy tensors if they do not exist,
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0,
                input_ids.shape[1],
                dtype=torch.long,
                device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask
        input_ids = [cur_input_ids[cur_attention_mask]
                     for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask]
                  for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []

        cur_image_idx = 0
        for cur_input_ids, cur_labels in zip(input_ids, labels):
            cur_new_input_embeds, cur_new_labels, cur_image_idx = self._insert_multimodal_embeds_per_batch(
                image_features, cur_input_ids, cur_labels, cur_image_idx)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the
        # sequence longer
        tokenizer_model_max_length = getattr(
            self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length]
                                for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Padding
        new_input_embeds_padded, new_labels_padded, attention_mask, position_ids = self._pad_multimodal_embeds_per_batch(
            new_input_embeds, new_labels, attention_mask, position_ids)
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        new_labels = None if _labels is None else new_labels_padded
        attention_mask = None if _attention_mask is None else attention_mask.to(
            dtype=_attention_mask.dtype)
        position_ids = None if _position_ids is None else position_ids

        cache_position = None if (
            _attention_mask is None or cache_position is None) else torch.arange(
            attention_mask.shape[1], device=attention_mask.device)

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, cache_position

    def _insert_multimodal_embeds_per_batch(
            self, image_features, cur_input_ids, cur_labels, cur_image_idx):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        if num_images == 0:
            # 用来跳过填充的空图像
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = self.backbone.embed_tokens(cur_input_ids)
            cur_input_embeds = torch.cat(
                [cur_input_embeds_1, cur_image_features[0:0]], dim=0)
            cur_image_idx += 1

            return cur_input_embeds, cur_labels, cur_image_idx

        image_positions = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
        image_token_indices = [-1] + image_positions + [
            cur_input_ids.shape[0]]  # [-1, image_index, end]

        cur_input_ids_noim = []
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(
                cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
            cur_labels_noim.append(
                cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
        # [0 -> image_index] [image_index+1 -> end]

        split_sizes = [x.shape[0] for x in cur_labels_noim]
        cur_input_embeds = self.backbone.embed_tokens(torch.cat(cur_input_ids_noim))
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

        cur_new_input_embeds = []
        cur_new_labels = []

        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])

            if i < num_images:
                cur_image_features = image_features[cur_image_idx]
                cur_new_input_embeds.append(cur_image_features)
                cur_new_labels.append(
                    torch.full(
                        (cur_image_features.shape[0],
                         ),
                        IGNORE_INDEX,
                        device=cur_labels.device,
                        dtype=cur_labels.dtype))
                cur_image_idx += 1

        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)

        return cur_new_input_embeds, cur_new_labels, cur_image_idx

    def _pad_multimodal_embeds_per_batch(
            self, new_input_embeds, new_labels, attention_mask, position_ids):
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size,
             max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device)
        attention_mask = torch.zeros(
            (batch_size,
             max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device)
        position_ids = torch.zeros(
            (batch_size,
             max_len),
            dtype=position_ids.dtype,
            device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(
                zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]

            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (torch.zeros(
                            (max_len - cur_len,
                             cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device),
                            cur_new_embed),
                        dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (cur_new_embed,
                         torch.zeros(
                             (max_len - cur_len,
                              cur_new_embed.shape[1]),
                             dtype=cur_new_embed.dtype,
                             device=cur_new_embed.device)),
                        dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        return new_input_embeds_padded, new_labels_padded, attention_mask, position_ids

    def _prepare_inputs_labels_for_multimodal_decode(self,
                                                     input_ids: torch.Tensor,
                                                     position_ids: torch.Tensor,
                                                     attention_mask: torch.Tensor,
                                                     past_key_values: torch.Tensor,
                                                     labels: torch.Tensor,
                                                     cache_position: torch.Tensor,
                                                     images: torch.Tensor) -> tuple:
        first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]
        batch_index, non_attended_tokens = torch.where(
            first_layer_past_key_value.float().sum(-2) == 0)

        target_length = input_ids.shape[1]  # the value should be 1
        past_length = first_layer_past_key_value.shape[-1]

        extended_attention_mask = torch.ones(
            (attention_mask.shape[0],
             past_length),
            dtype=attention_mask.dtype,
            device=attention_mask.device)

        # Filter out only the tokens that can be un-attended, this can happen
        # if one uses Llava + Fused modules where the cache on the
        # first iteration is already big enough, or if one passes custom cache
        valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
        new_batch_index = batch_index[valid_indices]
        new_non_attended_tokens = non_attended_tokens[valid_indices]

        # Zero-out the places where we don't need to attend
        extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

        attention_mask = torch.cat(
            (extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
        position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
        cache_position = torch.arange(
            attention_mask.shape[1], device=attention_mask.device)[-target_length:]

        return input_ids, position_ids, attention_mask, past_key_values, None, labels, cache_position


class DexboticForCausalLM(DexboticPretrainedModel, GenerationMixin):
    config_class = DexboticConfig

    def __init__(self, config: DexboticConfig):
        super().__init__(config)
        config.model_type = self.config_class.model_type

        self._real_init(config)

    def _real_init(self, config: DexboticConfig):
        self.model = DexboticVLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                images: Optional[torch.FloatTensor] = None,
                return_dict: Optional[bool] = None,
                cache_position: Optional[torch.LongTensor] = None,
                actions: Optional[torch.LongTensor] = None,
                states: Optional[torch.LongTensor] = None,
                ) -> CausalLMOutputDexbotic:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # TODO: output_hidden_states is not used actually
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
            cache_position
        ) = self.model._prepare_inputs_labels_for_multimodal(
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            cache_position,
            images
        )
        outputs = self.model.backbone(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict
        )

        hidden_states = outputs.hidden_states[-1]
        logits = self.lm_head(hidden_states)

        loss = None

        if labels is not None:
            loss = self.loss_function(logits, labels, self.model.backbone.vocab_size)

        return CausalLMOutputDexbotic(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def process_images(self, images):
        vision_tower = self.model.mm_vision_module
        image_processor = vision_tower.image_processor
        image_aspect_ratio = getattr(self.config, "image_aspect_ratio", 'pad')
        new_images = []
        if image_aspect_ratio == 'pad':
            for image in images:
                image = self.expand2square(image, tuple(int(x * 255)
                                           for x in image_processor.image_mean))
                image = image_processor.preprocess(
                    image, return_tensors='pt')['pixel_values'][0]
                new_images.append(image)
        else:
            return image_processor(images, return_tensors='pt')['pixel_values']
        if all(x.shape == new_images[0].shape for x in new_images):
            new_images = torch.stack(new_images, dim=0)
        return new_images

    @staticmethod
    def expand2square(pil_img, background_color):
        from PIL import Image
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None,
                                      **kwargs):
        images = kwargs.pop("images", None)

        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask, inputs_embeds=inputs_embeds,
            **kwargs
        )

        if images is not None:
            _inputs['images'] = images
        return _inputs
    


class ActionOutputForCausalLM(ABC):

    @abstractmethod
    def inference_action(self, input_ids, image_tensor, inference_args={}, **kwargs):
        ...

    def _denorm(self, actions, action_norms) -> np.ndarray:
        """Denormalize the actions
        Args:
            actions (np.array): Normalized actions with shape [T, D]
            action_norms (dict): Dictionary of normalization parameters
        """
        actions = np.clip(actions, -1, 1)
        min, max = np.array(action_norms['min']), np.array(action_norms['max'])
        min = min.reshape(1, -1)
        max = max.reshape(1, -1)
        actions = min + (actions + 1) * 0.5 * (max - min)
        return actions
