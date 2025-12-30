from typing import List, Optional
import copy
import torch
import torch.nn as nn
from dexbotic.model.modules.mm_vision.builder import build_vision_tower
from dexbotic.model.modules.mm_projector.builder import build_vision_projector
from dexbotic.model.dexbotic_arch import (ActionOutputForCausalLM,
                                          CausalLMOutputDexbotic,
                                          DexboticConfig, DexboticForCausalLM,
                                          DexboticVLMModel)
from transformers import (AutoConfig, AutoModel, PretrainedConfig,
                          PreTrainedModel,GenerationMixin)
from dexbotic.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from deepspeed.utils import safe_get_full_fp32_param

class MUVLAConfig(DexboticConfig):
    model_type = "dexbotic"
    action_model_type: Optional[str] = None
    action_dim: Optional[int] = None
    chunk_size: Optional[int] = None
    m_projector_type: Optional[str] = "mlp2x_gelu"
    mm_vision_tower: Optional[str] = None
    obs_vision_tower: Optional[str] = None
    chat_template: Optional[str] = "dexbotic"
    init_llm_weights: Optional[bool] = False

class CrossFuseReduce(nn.Module):
    def __init__(self, inter_dim=1024, fuse_len=1):
        super().__init__()
        self.reduce_proj = nn.Linear(4096, inter_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=inter_dim,
            num_heads=inter_dim // 64,  
            batch_first=True)

        self.ln = nn.LayerNorm(inter_dim)
        self.fuse_len = fuse_len
        if fuse_len > 1:
            self.proj = nn.Linear(inter_dim, fuse_len * inter_dim)

        self.back_proj = nn.Linear(inter_dim, 4096)

    def forward(self, map_tk, obs_tk):

        fused, _ = self.cross_attn(query=obs_tk, key=map_tk, value=map_tk)
        fused = self.ln(fused + obs_tk)
        return fused

class SimpleQFormer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_queries):
        super().__init__()
        self.query_embeddings = nn.Parameter(torch.randn(num_queries, hidden_dim))
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, vision_feats):
        B, T, D = vision_feats.shape
        memory = self.input_proj(vision_feats)  
        queries = self.query_embeddings.unsqueeze(0).expand(B, -1, -1)  
        out, _ = self.attn(query=queries, key=memory, value=memory)
        out = self.norm(out)
        return out



class MUVLAModel(DexboticVLMModel,GenerationMixin):
    def __init__(self, config: MUVLAConfig):
        super().__init__(config)

        if getattr(config, "init_llm_weights", False):
            self.llm = AutoModel.from_pretrained(config.llm_config)
            self.config.init_llm_weights = False
        else:
            llm_config = AutoConfig.from_pretrained(config.llm_config)
            self.llm = AutoModel.from_config(llm_config)

        self._merge_llm()

        self.mm_vision_tower = self._build_mm_vision_module(config.mm_vision_tower)
        self.obs_vision_tower = self._build_obs_vision_module(config.obs_vision_tower)

        self.mm_projector = self._build_mm_projector_module(config)
        self.fuser = self._build_fuser(config)
        self.history_qformer = self._build_history_qformer(config)

        self.post_init()


    @property
    def action_head_module(self) -> nn.Module:
        return self.action_head

    @property
    def action_head_prefix(self) -> str:
        return 'action_head'

    def initialize_model(self, extra_config: dict):
        for key, value in extra_config.items():
            setattr(self.config, key, value)
        self.mm_vision_tower = self._build_mm_vision_module(self.config.mm_vision_tower)
        self.obs_vision_tower = self._build_obs_vision_module(self.config.obs_vision_tower)
        self.mm_projector = self._build_mm_projector_module(self.config)
        self.fuser = self._build_fuser(self.config)
        self.history_qformer = self._build_history_qformer(self.config)

    def _merge_llm(self):
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

        self.mm_vision_tower = build_vision_tower(config)
        self.config.mm_hidden_size = self.mm_vision_tower.hidden_size

        return self.mm_vision_tower

    def _build_obs_vision_module(self, config) -> nn.Module:
        if getattr(self, 'obs_vision_tower', None) is not None:
            return self.obs_vision_tower

        self.obs_vision_tower = build_vision_tower(config)
        self.config.obs_hidden_size = self.obs_vision_tower.hidden_size

        return self.obs_vision_tower
    
    def _build_fuser(self, config) -> nn.Module:
        if getattr(self, 'fuser', None) is not None:
            return self.fuser
        self.fuser = CrossFuseReduce(inter_dim=1024, fuse_len=1)
        return self.fuser
    
    def _build_history_qformer(self, config) -> nn.Module:
        if getattr(self, 'history_qformer', None) is not None:
            return self.history_qformer
        self.history_qformer = SimpleQFormer(input_dim=1024, hidden_dim=1024,num_queries=192)
        return self.history_qformer
        
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
    def backbone(self):
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
            concat_images = torch.cat([image for image in images], dim=0)
            concat_image_features = encode_image(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(
                concat_image_features,
                split_sizes,
                dim=0) 
            image_features = [x.flatten(0, 1) for x in image_features]
            image_features = torch.stack(
                image_features, dim=0)  
        else:
            image_features = encode_image(images)

        image_features = image_features.to(self.device)
        return image_features

    def fuse_obs_with_history_and_project(self, map_img, obs_imgs):
        B, N, C, H, W = obs_imgs.shape
        obs_current = obs_imgs[:, 0, :, :, :]
        obs_history = obs_imgs[:, 1:, :, :, :]
        obs_tower = self.obs_vision_tower
        map_tower = self.mm_vision_tower
        map_projector = self.mm_projector
        fuser = self.fuser
        qformer = self.history_qformer
        obs_current_feat = obs_tower(obs_current)  
        if obs_history.size(1) == 0:
            obs_fused_feat = obs_current_feat
        else:
            T_hist = obs_history.size(1)
            obs_history = obs_history.reshape(B * T_hist, C, H, W)
            obs_hist_feat = obs_tower(obs_history)  
            obs_hist_feat = obs_hist_feat.reshape(B, T_hist * 576, -1)  
            qformer_feat = qformer(obs_hist_feat) 
            obs_fused_feat = torch.cat([qformer_feat, obs_current_feat], dim=1) 
        map_feat = map_tower(map_img)
        fused = fuser(map_feat, obs_fused_feat)
        projected = map_projector(fused)

        return projected

    def _prepare_inputs_labels_for_multimodal(self,
                                              input_ids: Optional[torch.Tensor],
                                              position_ids: Optional[torch.Tensor],
                                              attention_mask: Optional[torch.Tensor],
                                              past_key_values: Optional[torch.Tensor],
                                              labels: Optional[torch.Tensor],
                                              cache_position: Optional[torch.Tensor],
                                              images: Optional[torch.Tensor]) -> tuple:

        if input_ids.shape[1] == 1:
            return self._prepare_inputs_labels_for_multimodal_decode(
                input_ids, position_ids, attention_mask, past_key_values, labels, cache_position, images)

        vision = self.mm_vision_module

        if vision is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, cache_position

        if images.ndim == 5: 
            map_image = images[:, 0, :, :, :]
            obs_image = images[:, 1:, :, :, :]
            image_features = self.fuse_obs_with_history_and_project(map_image, obs_image)
            image_features = [feat.to(self.device) for feat in image_features]
        else:
            image_features = self._extract_vision_features(images)

        _labels, _position_ids, _attention_mask = labels, position_ids, attention_mask

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

        tokenizer_model_max_length = getattr(
            self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length]
                                for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

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
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = self.backbone.embed_tokens(cur_input_ids)
            cur_input_embeds = torch.cat(
                [cur_input_embeds_1, cur_image_features[0:0]], dim=0)
            cur_image_idx += 1

            return cur_input_embeds, cur_labels, cur_image_idx

        image_positions = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
        image_token_indices = [-1] + image_positions + [
            cur_input_ids.shape[0]]  

        cur_input_ids_noim = []
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(
                cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
            cur_labels_noim.append(
                cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])

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

        valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
        new_batch_index = batch_index[valid_indices]
        new_non_attended_tokens = non_attended_tokens[valid_indices]

        extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

        attention_mask = torch.cat(
            (extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
        position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
        cache_position = torch.arange(
            attention_mask.shape[1], device=attention_mask.device)[-target_length:]

        return input_ids, position_ids, attention_mask, past_key_values, None, labels, cache_position


class MUVLAForCausalLM(DexboticForCausalLM):
    config_class = MUVLAConfig

    def _real_init(self, config: MUVLAConfig):
        self.model = MUVLAModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.reward_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.post_init()
        self.config.is_encoder_decoder = False
        self.config.is_decoder = True

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
                repeated_diffusion_steps: int = 4,
                reward: Optional[torch.FloatTensor] = None,
                **kwargs,
                ) -> CausalLMOutputDexbotic:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        def _assert_finite(x, name):
            if x is None: 
                return
            if not torch.isfinite(x).all():

                with torch.no_grad():
                    msg = (f"{name} has NaN/Inf | "
                        f"min={x.min().item() if x.numel()>0 else 'n/a'}, "
                        f"max={x.max().item() if x.numel()>0 else 'n/a'}, "
                        f"shape={tuple(x.shape)}, dtype={x.dtype}")
                raise FloatingPointError(msg)
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
        _assert_finite(inputs_embeds, "inputs_embeds(after _prepare_inputs_labels_for_multimodal)")
        if attention_mask is not None:
            if attention_mask.dtype not in (torch.bool, torch.long, torch.int):
                attention_mask = attention_mask.to(torch.long)
            _assert_finite(attention_mask.float(), "attention_mask")
        outputs = self.model.llm(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=True,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        loss = None

        _assert_finite(hidden_states, "hidden_states(after llm)")
        _assert_finite(logits, "logits(after lm_head)")

        if labels is not None:
            tau = 0.7
            shift_logits = logits[..., :-1, :].contiguous() 
            shift_labels = labels[..., 1:].contiguous()     

            batch_size = shift_labels.shape[0]
            seq_len = shift_labels.shape[1]
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))  # [B*L]
            loss = loss.view(batch_size, seq_len)  # [B, L]
            mask = (shift_labels != -100).float()
            loss_per_sample = (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B]
            if reward is not None:
                weights = 1.0
                weights = 1.0 + torch.sigmoid(reward)  
                loss = (loss_per_sample * weights).mean()
            else:
                loss = loss_per_sample.mean()

        if reward is not None:
            reward = reward.to(hidden_states.dtype)
            reward_pred = self.reward_head(hidden_states).squeeze(-1)
            last_token_reward_pred = reward_pred[:, -1]
            reward_loss_fct = nn.MSELoss()
            reward_loss = reward_loss_fct(last_token_reward_pred, reward)
            diff = last_token_reward_pred - reward
            expectile = 0.9
            weight = torch.where(diff < 0, expectile, (1 - expectile))
            reward_loss = (weight * (diff**2)).mean()

            if loss is not None:
                loss += 0.5 * reward_loss
            else:
                loss = 0.2 * reward_loss


        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputDexbotic(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,

        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
       
        has_static_cache = False
        if past_key_values is None:
            past_key_values = getattr(getattr(self.model.llm.layers[0], "self_attn", {}), "past_key_value", None)
            has_static_cache = past_key_values is not None

        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, StaticCache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            elif isinstance(past_key_values, DynamicCache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = None   
            
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]

            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            else:
                remove_prefix_length = input_ids.shape[1] - 1
                input_ids = input_ids[:, remove_prefix_length:]
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:

            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        else:
            cache_position = cache_position[-input_length:]

        if has_static_cache:
            past_key_values = None

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "images": kwargs.get("images"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
