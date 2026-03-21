"""Higgs-Audio is an end-to-end multimodal model with the capability to understand and generate text / audio."""

import time
import torch
import torch.nn as nn
import math
import glob
import functools
import os
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from enum import Enum
from safetensors.torch import load_file
from typing import Optional, Tuple, Union, List, Dict, Any

import torch.nn.functional as F

from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LLAMA_ATTENTION_CLASSES,
    LlamaMLP,
    LlamaRMSNorm,
)
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin, GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.generation.utils import GenerateNonBeamOutput
from transformers.utils import logging, ModelOutput

from .common import HiggsAudioPreTrainedModel
from .utils import (
    merge_input_ids_with_audio_features,
    merge_input_ids_with_audio_features_without_audio_in_embed,
    count_parameters,
)
from .configuration_higgs_audio import HiggsAudioConfig, HiggsAudioEncoderConfig
from .custom_modules import PartiallyFrozenLinear, PartiallyFrozenEmbedding
from .cuda_graph_runner import CUDAGraphRunner
from .audio_head import HiggsAudioDecoderProjector

import torch.cuda.nvtx as nvtx

logger = logging.get_logger(__name__)


class GenerationMode(Enum):
    """Enum for different generation modes in HiggsAudio model."""

    TEXT = 0  # Text generation mode
    AUDIO_INIT = 1  # Audio generation mode initialization
    AUDIO_IN_PROGRESS = 2  # Audio generation mode in progress


def _whisper_encoder_zero_shape_forward(whisper_encoder, *args, **kwargs):
    """The whisper encoder does not support zero-shape tensor by default due to the following implementations

        key_states = self._shape(self.k_proj(current_states), -1, bsz)

    If `bsz` is 0, the "-1" dimension will be ambiguous and triggers error in the shape inference pass.

    See also: https://github.com/huggingface/transformers/blob/30335093276212ce74938bdfd85bfd5df31a668a/src/transformers/models/whisper/modeling_whisper.py#L306-L307

    This function monkey-patches all `_shape` functions in the whisper encoder's self-attention layers to ensure function supports zero-shape tensor.

    #FIXME!!!! This is a temporary workaround and should be removed once the upstream issue is resolved.

    """

    global _higgs_flash_attention_forward

    def _patched_shape(tensor: torch.Tensor, seq_len: int, bsz: int, num_heads: int, head_dim: int):
        if seq_len == -1:
            return tensor.view(bsz, tensor.shape[1], num_heads, head_dim).transpose(1, 2).contiguous()
        else:
            return tensor.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()

    def _patched_scaled_dot_product_attention(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
    ) -> torch.Tensor:
        # IMPORTANT! Implementation here is wrong and is only for the purpose of obtaining the correct attn_weight shape
        if enable_gqa:
            key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1)
        return attn_weight @ value

    # Apply monkey-patch
    if whisper_encoder.config._attn_implementation != "flash_attention_2":
        old_shape_functions = []
        for layer in whisper_encoder.layers:
            old_shape_functions.append(getattr(layer.self_attn, "_shape"))
            layer.self_attn._shape = functools.partial(
                _patched_shape, num_heads=layer.self_attn.num_heads, head_dim=layer.self_attn.head_dim
            )

    original_scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
    torch.nn.functional.scaled_dot_product_attention = _patched_scaled_dot_product_attention

    out = whisper_encoder(*args, **kwargs)
    torch.nn.functional.scaled_dot_product_attention = original_scaled_dot_product_attention

    # Restore the original shape functions
    if whisper_encoder.config._attn_implementation != "flash_attention_2":
        for layer, old_shape_function in zip(whisper_encoder.layers, old_shape_functions):
            layer.self_attn._shape = old_shape_function

    return out


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask


def create_interleaved_modal_mask(
    audio_out_mask: torch.BoolTensor,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    """
    生成符合新规则的4D注意力掩码：
    - 文本query：仅关注文本key（屏蔽音频key）
    - 音频query：关注文本key + 音频key（均需因果性）
    输入：
      - attention_mask=None：自动生成4D因果掩码，再叠加模态规则
      - attention_mask≠None：使用输入的4D因果掩码（B,1,Q,K），叠加模态规则
    输出：4D掩码（B,1,Q,K）
    """
    # 基础参数提取
    batch_size = hidden_states.shape[0]
    target_length = hidden_states.shape[1]  # Q：当前query长度
    seq_len = audio_out_mask.shape[1]       # K：总key长度
    dtype = hidden_states.dtype
    device = hidden_states.device
    min_dtype = torch.finfo(dtype).min

    # 1. 处理cache_position（默认时序索引）
    if cache_position is None:
        cache_position = torch.arange(target_length, device=device)

    # 2. 生成基础4D因果掩码（若attention_mask为None）
    if attention_mask is None:
        # 2.1 先生成2D因果掩码（Q×K）
        causal_mask = torch.full((target_length, seq_len), fill_value=min_dtype, dtype=dtype, device=device)
        if target_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)  # 上三角为min_dtype（屏蔽未来）
            # 叠加缓存位置过滤
            causal_mask *= torch.arange(seq_len, device=device) > cache_position.reshape(-1, 1)
        # 2.2 扩展为4D（B,1,Q,K）
        attention_mask = causal_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, -1, -1)

    # 此时attention_mask必为4D（B,1,Q,K），已包含因果性和padding处理

    # 3. 标记文本/音频的query和key（模态区分）
    # 3.1 Query模态：当前处理的前target_length个位置（Q维度）
    query_audio_mask = audio_out_mask[:, :target_length].bool()  # (B, Q)：True=音频query
    query_text_mask = ~query_audio_mask                          # (B, Q)：True=文本query
    # 3.2 Key模态：全序列位置（K维度）
    key_audio_mask = audio_out_mask.bool()                       # (B, K)：True=音频key
    key_text_mask = ~key_audio_mask                              # (B, K)：True=文本key（含输入/输出文本）

    # 4. 扩展模态掩码为4D，适配attention_mask的广播
    query_text_4d = query_text_mask.unsqueeze(1).unsqueeze(3)  # (B,1,Q,1)：文本query位置
    key_audio_4d = key_audio_mask.unsqueeze(1).unsqueeze(2)    # (B,1,1,K)：音频key位置

    # 5. 生成「需屏蔽的位置」：仅文本query关注音频key的情况（其他情况均允许）
    # 规则：文本query + 音频key → 屏蔽（设为min_dtype）
    # 允许：文本query+文本key、音频query+文本key、音频query+音频key
    text_query_audio_key_mask = query_text_4d & key_audio_4d  # (B,1,Q,K)：True=需屏蔽

    # 6. 在已有因果掩码上叠加屏蔽：仅屏蔽文本query→音频key的位置
    final_mask = attention_mask.clone()  # 复制原有因果掩码
    final_mask.masked_fill_(text_query_audio_key_mask, min_dtype)  # 执行屏蔽

    return final_mask


def create_interleaved_modal_mask_complex(
    audio_out_mask: torch.BoolTensor,  # 原有：A位置=True，其他=False
    input_ids: torch.LongTensor,       # 新增：输入序列的token ID，用于识别<audio_out_bos>/<audio_out_last_bos_id>/<audio_eos>
    audio_special_ids: Tuple[int, int],# 新增：(audio_out_bos_id, audio_out_last_bos_id, audio_eos_id)，固定标记的ID
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    """
    仅用audio_out_mask+input_ids推导所有模态，实现注意力规则：
    - T（纯文本）：仅关注自身及前面的T
    - 音频相关文本（<bos>/<eos>）：关注T + 自身及前面的音频相关文本，屏蔽A
    - A（音频）：关注T + 音频相关文本 + 自身，屏蔽其他A
    """
    # 基础参数提取
    batch_size = hidden_states.shape[0]
    target_length = hidden_states.shape[1]  # 当前query长度（Q）

    pre_target_length = target_length
    if target_length < 5 and cache_position is not None and cache_position.shape[-1] < 5:
        target_length = input_ids.shape[-1]
    if cache_position is not None:
        audio_out_mask = input_ids == 128016
    seq_len = audio_out_mask.shape[1]  # 总key长度（K）
    dtype = hidden_states.dtype
    device = hidden_states.device
    min_dtype = torch.finfo(dtype).min
    audio_out_bos_id, audio_out_last_bos_id, audio_eos_id = audio_special_ids  # 解包音频相关标记的ID

    # 1. 处理cache_position（默认时序索引）
    if cache_position is None:
        cache_position = torch.arange(target_length, device=device)

    # 2. 生成基础4D因果掩码（若attention_mask为None）
    if attention_mask is None:
        causal_mask = torch.full((target_length, seq_len), fill_value=min_dtype, dtype=dtype, device=device)
        if target_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)  # 屏蔽未来位置
            causal_mask *= torch.arange(seq_len, device=device) > cache_position.reshape(-1, 1)  # 缓存过滤
        attention_mask = causal_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, -1, -1)  # 扩展为4D

    # 3. 核心：通过input_ids+逻辑运算推导三种模态mask（无需额外传参）
    # 3.1 音频相关文本mask：input_ids等于<audio_out_bos>或<audio_eos>的位置
    audio_text_mask = (input_ids == audio_out_bos_id) | (input_ids == audio_out_last_bos_id) | (input_ids == audio_eos_id)  # (B, seq_len)
    # 3.2 纯文本T mask：非A（~audio_out_mask）且 非音频相关文本（~audio_text_mask）
    t_mask = ~audio_out_mask & ~audio_text_mask  # (B, seq_len)
    # 3.3 音频A mask：直接用原有audio_out_mask
    a_mask = audio_out_mask  # (B, seq_len)

    # 4. 标记当前query的模态（仅取前target_length个位置，即当前处理的query）
    query_t_mask = t_mask[:, :target_length].bool()  # (B, Q)：T模态query
    query_audio_text_mask = audio_text_mask[:, :target_length].bool()  # (B, Q)：音频相关文本query
    query_a_mask = a_mask[:, :target_length].bool()  # (B, Q)：A模态query

    # 5. 扩展query模态mask为4D（适配广播，与attention_mask形状对齐）
    query_t_4d = query_t_mask.unsqueeze(1).unsqueeze(3)  # (B,1,Q,1)
    query_audio_text_4d = query_audio_text_mask.unsqueeze(1).unsqueeze(3)  # (B,1,Q,1)
    query_a_4d = query_a_mask.unsqueeze(1).unsqueeze(3)  # (B,1,Q,1)

    # 6. 标记全序列key的模态（扩展为4D）
    key_t_4d = t_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,K)：T模态key
    key_audio_text_4d = audio_text_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,K)：音频相关文本key
    key_a_4d = a_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,K)：A模态key

    # 7. 生成“A仅关注自身”的mask：仅当key索引=query索引且为A时允许
    query_indices = torch.arange(target_length, device=device).reshape(1, 1, target_length, 1)  # (1,1,Q,1)
    key_indices = torch.arange(seq_len, device=device).reshape(1, 1, 1, seq_len)  # (1,1,1,K)
    key_self_a_4d = (query_indices == key_indices) & key_a_4d  # (B,1,Q,K)：仅自身A允许

    # 8. 定义各模态query的“允许关注规则”
    # 8.1 T模态query：仅允许关注T模态key（因果内）
    t_allow = query_t_4d & key_t_4d
    # 8.2 音频相关文本query：允许关注T + 音频相关文本key（因果内）
    audio_text_allow = query_audio_text_4d & (key_t_4d | key_audio_text_4d)
    # 8.3 A模态query：允许关注T + 音频相关文本 + 自身A（因果内）
    # if str(device) == 'cuda:7':
    #     print(key_t_4d.shape, 'key_t_4d\n')
    #     print(key_audio_text_4d.shape, 'key_audio_text_4d\n')
    #     print(key_self_a_4d.shape, 'key_self_a_4d\n')
    #     print(query_a_4d.shape, 'query_a_4d')

    # a_allow = query_a_4d & (key_t_4d | key_audio_text_4d | key_self_a_4d)
    a_allow = query_a_4d & (key_t_4d | key_audio_text_4d | key_a_4d)

    # 9. 总允许矩阵：合并三种query的允许规则
    total_allow = t_allow | audio_text_allow | a_allow

    # 10. 生成最终掩码：屏蔽不允许的位置
    audio_attention_mask = attention_mask.clone()
    text_attention_mask = attention_mask.clone()
    # if str(device) == 'cuda:7':
    #     print(text_attention_mask.shape, 'text_attention_mask.shape\n')
    #     print(total_allow.shape, 'total_allow.shape\n')
    text_attention_mask[..., :total_allow.shape[-1]].masked_fill_(~total_allow[..., -text_attention_mask.shape[2]:, :], min_dtype)  # ~total_allow：不允许关注的位置

    if cache_position is not None and cache_position.shape[-1] < 5:
        text_attention_mask = text_attention_mask[:, :, -pre_target_length:, :]
        audio_attention_mask = audio_attention_mask[:, :, -pre_target_length:, :]

    return text_attention_mask, audio_attention_mask


class HiggsAudioFeatureProjector(nn.Module):
    """Projector that maps audio features extracted by Whisper to hidden state of the text model."""

    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.linear = nn.Linear(config.audio_encoder_config.d_model, config.text_config.hidden_size, bias=True)

    def forward(self, audio_features):
        hidden_states = self.linear(audio_features)
        return hidden_states


class AudioAligner(nn.Module):
    """Projector that maps audio features extracted by Whisper to hidden state of the text model. """
    """
    'model.mm_audio_aligner.projector.0.bias', 
    'model.mm_audio_aligner.projector.0.weight', 
    'model.mm_audio_aligner.projector.2.bias', 
    'model.mm_audio_aligner.projector.2.weight'
    """
    def __init__(self, audio_encoder_dim=1280, text_hidden_dim=3072):
        super().__init__()
        mlp_depth = 2
        modules = []
        modules.append(nn.Linear(audio_encoder_dim, text_hidden_dim, bias=True))
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(text_hidden_dim, text_hidden_dim, bias=True))
        projector = nn.Sequential(*modules)
        self.projector = projector

    def forward(self, audio_features):
        projector_output = self.projector(audio_features)
        return projector_output

# Revised on top of transformers.models.qwen2_audio.modeling_qwen2_audio with Qwen2AudioEncoder --> HiggsAudioEncoder
# The code was originally borrowed from WhisperEncoder
class HiggsAudioEncoder(HiggsAudioPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`WhisperEncoderLayer`].

    Args:
        config: HiggsAudioEncoderConfig
    """

    # Ignore copy
    config_class = HiggsAudioEncoderConfig
    main_input_name = "input_features"
    _no_split_modules = ["WhisperEncoderLayer"]

    def __init__(self, config: HiggsAudioEncoderConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)

        # Flash Attention 2 does not support zero shape tensor, so we have to use sdpa implementation for the Whisper component.
        self.layers = nn.ModuleList([WhisperEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)
        # Ignore copy
        self.avg_pooler = nn.AvgPool1d(2, stride=2)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        check_seq_length=True,
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`torch.Tensor`)`, *optional*):
                HiggsAudio does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """

        expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        if check_seq_length and (input_features.shape[-1] != expected_seq_length):
            raise ValueError(
                f"HiggsAudio expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Ignore copy
        input_features = input_features.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (len(self.layers)), (
                f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
            )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            # Ignore copy
            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Ignore copy
        hidden_states = hidden_states.permute(0, 2, 1)
        # If the sequence length after average pooling is not divisible by the sequence parallel size, we would duplicate it across the sequence parallel ranks.
        # In this case, gradients need to be scaled up because the subsequent scaling up in the function _apply_audio_tower is skipped.
        hidden_states = self.avg_pooler(hidden_states)

        hidden_states = hidden_states.permute(0, 2, 1)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

    # Ignore copy
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths


class HiggsAudioDualFFNDecoderLayer(nn.Module):
    """We implement a dual-path FFN decoder layer where the audio tokens and text tokens go through separate FFN layers.

    The audio and text tokens share the text-attention layer, but will be encoded with separate feedforward layers.
    In addition, the audio tokens can be configured to go through separate attention layer.

    Following is an illustration:

     t    t    t    a   a     a    t    t    t
                        |
                        | (shared attention layer)
                        v
    h_t  h_t  h_t  h_a  h_a  h_a  h_t  h_t  h_t
                        |
                        | (separate text/audio hidden states)
                        v
    [h_t  h_t  h_t  h_t  h_t  h_t], [h_a, h_a, h_a]
             |                             |
             | (separate FFNs)             |
             v                             v
    [o_t  o_t  o_t  o_t  o_t  o_t], [o_a, o_a, o_a]
                        |
                        | (reorder)
                        v
    o_t  o_t  o_t  o_a  o_a  o_a  o_t  o_t  o_t

    This has a few advantages:
    1) We are able to use a smaller FFN, or even bypass the FFN for audio tokens. This accelerates the inference speed.
    2) The Audio-FFN introduces more trainable parameters to the model.
       This should have the same effect as the mixture-of-expert layer and we may expect better performance due to parameter scaling.
    3) We can replace the original FFN in LLMs with the dual-path FFN without changing the number of FLOPs.


    """

    def __init__(
        self, config: HiggsAudioConfig, layer_idx: int, fast_forward: bool = False, use_audio_attention: bool = False
    ):
        super().__init__()
        text_config = config.text_config
        self.hidden_size = text_config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=text_config, layer_idx=layer_idx)

        # print(config._attn_implementation, 'config._attn_implementation')
        # print(text_config._attn_implementation, 'text_config._attn_implementation')
        # print(type(self.self_attn), 'type(self.self_attn)')

        self.mlp = LlamaMLP(text_config)

        if not fast_forward:
            audio_config = text_config
            audio_config.intermediate_size = 8192  # 8192

            if use_audio_attention:
                self.audio_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](
                    config=audio_config, layer_idx=layer_idx + 1
                )
                self.audio_post_audio_attn_layer_norm = LlamaRMSNorm(
                    audio_config.hidden_size, eps=audio_config.rms_norm_eps
                )

            self.audio_mlp = LlamaMLP(audio_config)
            self.audio_input_layernorm = LlamaRMSNorm(audio_config.hidden_size, eps=audio_config.rms_norm_eps)
            self.audio_post_attention_layernorm = LlamaRMSNorm(audio_config.hidden_size, eps=audio_config.rms_norm_eps)

            # if use_audio_attention:
            #     self.audio_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](
            #         config=text_config, layer_idx=layer_idx + 1
            #     )
            #     self.audio_post_audio_attn_layer_norm = LlamaRMSNorm(
            #         text_config.hidden_size, eps=text_config.rms_norm_eps
            #     )

            # self.audio_mlp = LlamaMLP(text_config)
            # self.audio_input_layernorm = LlamaRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
            # self.audio_post_attention_layernorm = LlamaRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

        self.use_audio_attention = use_audio_attention
        self.fast_forward = fast_forward
        if self.fast_forward:
            assert not self.use_audio_attention, (
                "We cannot use audio_attention if the layer is marked as fast-forward."
            )
        self.input_layernorm = LlamaRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        fast_forward_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        audio_out_mask: Optional[torch.BoolTensor] = None,
        is_decoding_audio_token: Optional[bool] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        is_using_cuda_graph: Optional[bool] = False,
        **kwargs,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            position_ids
                IDs of positions in the input sequence
            audio_out_mask
                Mask for identifying the audio tokens. Size (batch_size, sequence_length)
                1 --> location contains audio_out
                0 --> location does not contain audio_out

                When use_cache is True and not in torch compile mode, the audio_out_mask contains audio_out masks for
                all tokens up to the current token.  That means, it has size (batch_size, sequence_length) while
                hidden_states will have size (batch_size, 1). In the torch compile mode, the audio_out_mask will have
                size (batch_size, 1).
            is_decoding_audio_token
                Used in the torch compile mode to determine if the current token is an audio token or not.
            past_key_value (`Cache`, *optional*): cached past key and value projection states. We fetch the corresponding cached key/value via the layer_idx.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            is_using_cuda_graph (`bool`, *optional*):
                Indicates whether the model is running by cuda graph.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states
        target_length = hidden_states.shape[1]
        use_static_cache = isinstance(past_key_value, StaticCache)
        decode_stage = hidden_states.shape[1] == 1
        if is_using_cuda_graph:
            assert decode_stage and use_static_cache, (
                "The CUDA graph mode should only be used in the decoding stage with static cache."
            )

        # If we are decoding an audio token and the layer is marked as fast-forward,
        # we can skip it.
        if is_decoding_audio_token and self.fast_forward:
            return (hidden_states,)

        has_audio_out = audio_out_mask is not None and audio_out_mask.shape[0] > 0

        audio_out_mask_sq = audio_out_mask

        # print(self.self_attn.config._attn_implementation, 'self.self_attn.config._attn_implementation')

        # if self.fast_forward and has_audio_out:
        #     original_hidden_states = hidden_states.clone()
        #     min_dtype = torch.finfo(hidden_states.dtype).min

        #     if attention_mask is None:
        #         attention_mask = ~audio_out_mask

        #         if self.self_attn.config._attn_implementation != "flash_attention_2":
        #             sequence_length = audio_out_mask.shape[1]
        #             attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
        #                 attention_mask=attention_mask,
        #                 sequence_length=sequence_length,
        #                 target_length=sequence_length,
        #                 dtype=hidden_states.dtype,
        #                 min_dtype=min_dtype,
        #                 device=hidden_states.device,
        #                 cache_position=cache_position,
        #                 batch_size=hidden_states.shape[0],
        #             )
        #             if use_cache:
        #                 attention_mask = attention_mask[:, :, -target_length:, :]
        #     elif len(attention_mask.shape) == 2:
        #         # Attention mask has shape (batch_size, sequence_length)
        #         # We should be using flash attention 2
        #         attention_mask = attention_mask * ~audio_out_mask
        #     elif len(attention_mask.shape) == 4:
        #         # When using static cache, the attention mask was already preprocessed in the previous layer
        #         if use_static_cache:
        #             attention_mask = fast_forward_attention_mask
        #         else:
        #             if use_cache:
        #                 # Attention mask has shape (batch_size, 1, query_length, key_length)
        #                 # In addition, the attention mask should be inverted, that means "1" (attend_to) --> "0", and "0" --> minimal dtype value.
        #                 attention_mask = attention_mask.masked_fill(
        #                     audio_out_mask[:, -target_length:].reshape(audio_out_mask.shape[0], 1, target_length, 1)
        #                     | audio_out_mask.reshape(audio_out_mask.shape[0], 1, 1, audio_out_mask.shape[1]),
        #                     min_dtype,
        #                 )
        #             else:
        #                 attention_mask = attention_mask.masked_fill(
        #                     audio_out_mask.reshape(audio_out_mask.shape[0], 1, audio_out_mask.shape[1], 1)
        #                     | audio_out_mask.reshape(audio_out_mask.shape[0], 1, 1, audio_out_mask.shape[1]),
        #                     min_dtype,
        #                 )
        #     else:
        #         raise NotImplementedError(f"Unsupported attention_mask format, attention_mask={attention_mask}")


        #     if (
        #         self.self_attn.config._attn_implementation == "sdpa"
        #         and attention_mask is not None
        #         and attention_mask.device.type == "cuda"
        #         and not output_attentions
        #     ):
        #         # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
        #         # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        #         # Details: https://github.com/pytorch/pytorch/issues/110213
        #         attention_mask = AttentionMaskConverter._unmask_unattended(attention_mask, min_dtype)
        if has_audio_out and not self.fast_forward:
            # Apply separate layernorm layers for audio tokens and text tokens
            if use_cache:
                hidden_states = torch.where(
                    audio_out_mask_sq[:, -target_length:].unsqueeze(-1),
                    self.audio_input_layernorm(hidden_states),
                    self.input_layernorm(hidden_states),
                )
            else:
                hidden_states = torch.where(
                    audio_out_mask_sq.unsqueeze(-1),
                    self.audio_input_layernorm(hidden_states),
                    self.input_layernorm(hidden_states),
                )
        else:
            hidden_states = self.input_layernorm(hidden_states)

        # # edit output_attentions
        # output_attentions = True

        # Text Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=text_attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )


        # # edit output_attentions
        # if cur_device == 'cuda:1':
        #     print('self_attn_weights', self_attn_weights, self_attn_weights.shape)
        #     for i in range(self_attn_weights.shape[2]):
        #         for j in range(self_attn_weights.shape[3]):
        #             if self_attn_weights[0][0][i][j] > 1e-7 and j > i:
        #                 print('text_attn error!!! see future', i, j, audio_out_mask[0][i], audio_out_mask[0][j])
        #             elif self_attn_weights[0][0][i][j] < 1e-8 and j <= i:
        #                 print('text_attn mask before', i, j, audio_out_mask[0][i], audio_out_mask[0][j])
        # output_attentions = False
        # import time
        # time.sleep(10000)


        hidden_states = residual + hidden_states
        residual = hidden_states
        # if use_cache:
        #     hidden_states = torch.where(
        #         ~audio_out_mask_sq[:, -target_length:].unsqueeze(-1), residual + hidden_states, hidden_states
        #     )
        # else:
        #     hidden_states = torch.where(~audio_out_mask_sq.unsqueeze(-1), residual + hidden_states, hidden_states)

        # Audio Attention
        if self.use_audio_attention and has_audio_out:
            if use_static_cache:
                assert audio_attention_mask is not None, (
                    "audio_attention_mask should not be None when using static cache."
                )

            # if audio_attention_mask is None:
            #     no_audio_out_mask = (~audio_out_mask)[:, -target_length:].reshape(
            #         audio_out_mask.shape[0], 1, target_length, 1
            #     ) | (~audio_out_mask).reshape(audio_out_mask.shape[0], 1, 1, audio_out_mask.shape[1])
            #     min_dtype = torch.finfo(hidden_states.dtype).min

            #     if attention_mask is None:
            #         audio_attention_mask = audio_out_mask

            #         if self.audio_attn.config._attn_implementation != "flash_attention_2":
            #             sequence_length = audio_out_mask.shape[1]
            #             audio_attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            #                 attention_mask=audio_attention_mask,
            #                 sequence_length=sequence_length,
            #                 target_length=sequence_length,
            #                 dtype=hidden_states.dtype,
            #                 min_dtype=min_dtype,
            #                 device=hidden_states.device,
            #                 cache_position=cache_position,
            #                 batch_size=hidden_states.shape[0],
            #             )
            #             if use_cache:
            #                 audio_attention_mask = audio_attention_mask[:, :, -target_length:, :]
            #             audio_attention_mask = audio_attention_mask.masked_fill(no_audio_out_mask, min_dtype)
            #     elif len(attention_mask.shape) == 2:
            #         # Attention mask has shape (batch_size, sequence_length)
            #         audio_attention_mask = attention_mask * audio_out_mask
            #     elif len(attention_mask.shape) == 4:
            #         # Attention mask has shape (batch_size, 1, query_length, key_length)
            #         # In addition, the attention mask should be inverted. This means "1" (attend_to) --> "0", and "0" --> minimal dtype value.
            #         audio_attention_mask = attention_mask.masked_fill(no_audio_out_mask, min_dtype)
            #     else:
            #         raise NotImplementedError(f"Unsupported attention_mask format, attention_mask={attention_mask}")

            #     if (
            #         self.audio_attn.config._attn_implementation == "sdpa"
            #         and audio_attention_mask is not None
            #         and audio_attention_mask.device.type == "cuda"
            #         and not output_attentions
            #     ):
            #         # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            #         # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            #         # Details: https://github.com/pytorch/pytorch/issues/110213
            #         audio_attention_mask = AttentionMaskConverter._unmask_unattended(audio_attention_mask, min_dtype)

            audio_attention_mask = audio_attention_mask.contiguous()


            # # edit output_attentions
            # output_attentions = True

            audio_hidden_states = self.audio_post_audio_attn_layer_norm(hidden_states)

            audio_hidden_states, audio_self_attn_weights, audio_present_key_value = self.audio_attn(
                hidden_states=audio_hidden_states,
                attention_mask=audio_attention_mask,  # audio_attention_mask
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )


            # # edit output_attentions
            # cur_device = str(audio_self_attn_weights.device)
            # if cur_device == 'cuda:1':
            #     print('audio_self_attn_weights', audio_self_attn_weights, audio_self_attn_weights.shape)
            #     for i in range(audio_self_attn_weights.shape[2]):
            #         for j in range(audio_self_attn_weights.shape[3]):
            #             if audio_self_attn_weights[0][0][i][j] > 1e-7 and j > i:
            #                 print('audio_attn error!!! see future', i, j, audio_out_mask[0][i], audio_out_mask[0][j])
            #             elif audio_self_attn_weights[0][0][i][j] < 1e-8 and j <= i:
            #                 print('audio_attn error!!! mask before', i, j, audio_out_mask[0][i], audio_out_mask[0][j])
            # output_attentions = False


            audio_hidden_states = residual + audio_hidden_states
            # if use_cache:
            #     residual = torch.where(
            #         audio_out_mask_sq[:, -target_length:].unsqueeze(-1), audio_hidden_states, residual
            #     )
            # else:
            #     residual = torch.where(audio_out_mask_sq.unsqueeze(-1), audio_hidden_states, residual)
            # audio_hidden_states = self.audio_post_audio_attn_layer_norm(audio_hidden_states)
            if use_cache:
                hidden_states = torch.where(
                    audio_out_mask_sq[:, -target_length:].unsqueeze(-1), audio_hidden_states, hidden_states
                )
            else:
                hidden_states = torch.where(audio_out_mask_sq.unsqueeze(-1), audio_hidden_states, hidden_states)


        # Apply Dual-path FFN
        residual = hidden_states

        if has_audio_out and not self.fast_forward:
            if use_cache:
                real_audio_out_mask = audio_out_mask_sq[:, -target_length:]
            else:
                real_audio_out_mask = audio_out_mask_sq

            # Make whole graph in decode stage
            if decode_stage and is_using_cuda_graph:
                assert is_decoding_audio_token is not None, (
                    "is_decoding_audio_token should be present in the decoding stage."
                )
                if is_decoding_audio_token:
                    hidden_states = self.audio_post_attention_layernorm(hidden_states)
                    hidden_states = self.audio_mlp(hidden_states)
                else:
                    hidden_states = self.post_attention_layernorm(hidden_states)
                    hidden_states = self.mlp(hidden_states)
                residual = residual + hidden_states
            else:
                text_hidden_states = self.post_attention_layernorm(hidden_states[~real_audio_out_mask])
                audio_hidden_states = self.audio_post_attention_layernorm(hidden_states[real_audio_out_mask])

                text_hidden_states = self.mlp(text_hidden_states)
                residual[~real_audio_out_mask] = residual[~real_audio_out_mask] + text_hidden_states

                audio_hidden_states = self.audio_mlp(audio_hidden_states)
                residual[real_audio_out_mask] = residual[real_audio_out_mask] + audio_hidden_states

            hidden_states = residual
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

        # if self.fast_forward and has_audio_out:
        #     if use_cache:
        #         hidden_states = torch.where(
        #             audio_out_mask_sq[:, -target_length:].unsqueeze(-1), original_hidden_states, hidden_states
        #         )
        #     else:
        #         hidden_states = torch.where(audio_out_mask_sq.unsqueeze(-1), original_hidden_states, hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            if self.use_audio_attention:
                # The returned attn weights have shape (batch_size, num_heads + num_audio_attn_heads, seq_length, seq_length)
                outputs += (torch.concat([self_attn_weights, audio_self_attn_weights], dim=1),)
            else:
                # The returned attn weights have shape (batch_size, num_heads, seq_length, seq_length)
                outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


@dataclass
class HiggsAudioModelOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    llm_loss: Optional[torch.FloatTensor] = None
    audio_loss: Optional[torch.FloatTensor] = None
    codebook_losses: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    expanded_input_ids: Optional[torch.LongTensor] = None
    expanded_labels: Optional[torch.LongTensor] = None
    audio_in_mask: Optional[torch.BoolTensor] = None
    audio_in_discrete_codes_mask: Optional[torch.BoolTensor] = None
    audio_out_mask: Optional[torch.BoolTensor] = None
    attention_mask: Optional[torch.BoolTensor] = None
    audio_logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    audio_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class HiggsAudioGenerationOutput(ModelOutput):
    """
    Outputs of HiggsAudio generation models, when using non-beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        audio_sequences (`tuple(torch.LongTensor)` *optional*):
            The generated discrete audio codes. These codes can be used to fill-in related locations of <|AUDIO_OUT|> at input sequences.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token).
            If the generated token is a text token, the tensor will have shape `(batch_size, config.vocab_size)`.
            If the generated token is an audio token, the tensor will have shape `(config.audio_num_codebooks, self.audio_codebook_size)`
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`):
            Unprocessed prediction scores of the language modeling head or the audio head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token).
            If the generated token is a text token, the tensor will have shape `(batch_size, config.vocab_size)`.
            If the generated token is an audio token, the tensor will have shape `(config.audio_num_codebooks, self.audio_codebook_size)`
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True`):
            Returns the model cache, used to speed up decoding. Different models have a different cache format, check
            the model's documentation. Usually, a [`~cache_utils.Cache`] instance.
    """

    sequences: torch.LongTensor = None
    audio_sequences: Optional[List[torch.LongTensor]] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None


class HiggsAudioModel(HiggsAudioPreTrainedModel, GenerationMixin):
    """Higgs-Audio is an end-to-end multimodal model with the capability to understand and generate text / audio.

    Consider the following example for mixed text/audio understanding / generation:

    - input_tokens: <text_token1><|audio_bos|>[AUDIO]<|audio_eos|><text_token2><|audio_bos|>[AUDIO]<|audio_eos|><text_token4>
    - input_tokens: <text_token1><|audio_bos|>[AUDIO]<|audio_eos|><text_token2><|audio_out_bos|>[AUDIO_OUT]<|audio_eos|><text_token4>

    We will fill [AUDIO] with the audio features extracted by Whisper and fill [AUDIO_OUT] with the audio tokens.

    Consider the following example for mixed text/audio generation:

    text: <|audio_out_bos|>    MASK           MASK           MASK          MASK               MASK         <|audio_eos|> [text_token1]
    audio:     MASK    <|audio_stream_bos|> [audio_token1] [audio_token2] [audio_token3] <|audio_stream_eos|>   MASK           MASK
    token_type: 0               1              1              1             1                  1                 0              0

    """

    _supports_cache_class = True
    _supports_static_cache = True

    def __init__(self, config: HiggsAudioConfig):
        super().__init__(config)

        self.padding_idx = config.pad_token_id
        self.audio_in_token_idx = config.audio_in_token_idx
        self.audio_out_token_idx = config.audio_out_token_idx
        self.audio_out_bos_token_id = config.audio_out_bos_token_id #if "audio_out_bos_token_id" in config else None
        self.audio_out_last_bos_token_id = 128030
        self.audio_eos_token_id = config.audio_eos_token_id #if "audio_eos_token_id" in config else None
        self.vocab_size = config.text_config.vocab_size
        self.audio_num_codebooks = config.audio_num_codebooks
        self.use_delay_pattern = config.use_delay_pattern
        self.use_audio_out_embed_projector = config.use_audio_out_embed_projector
        self.use_audio_out_self_attention = config.use_audio_out_self_attention

        self.embed_tokens = nn.Embedding(self.vocab_size, config.text_config.hidden_size, self.padding_idx)


        # edit merge_cache
        self.input_merge_cache = torch.zeros(
            1,  # 你的场景batch_size=1，固定为1
            4096,
            dtype=torch.long  # 和input_ids同 dtype
        )



        if config.audio_adapter_type == "dual_ffn":
            layer_idx = 0
            layers = []
            for j in range(config.text_config.num_hidden_layers):
                if j in config.audio_dual_ffn_layers:
                    layers.append(
                        HiggsAudioDualFFNDecoderLayer(
                            config, layer_idx, use_audio_attention=self.use_audio_out_self_attention
                        )
                    )
                    layer_idx += 2 if self.use_audio_out_self_attention else 1
                else:
                    layers.append(LlamaDecoderLayer(config.text_config, layer_idx))
                    layer_idx += 1
            self.layers = nn.ModuleList(layers)
        elif config.audio_adapter_type == "dual_ffn_fast_forward":
            layer_idx = 0
            layers = []
            for j in range(config.text_config.num_hidden_layers):
                if j in config.audio_dual_ffn_layers:
                    layers.append(
                        HiggsAudioDualFFNDecoderLayer(
                            config,
                            layer_idx,
                            fast_forward=False,
                            use_audio_attention=self.use_audio_out_self_attention,
                        )
                    )
                    layer_idx += 2 if self.use_audio_out_self_attention else 1
                else:
                    layers.append(
                        HiggsAudioDualFFNDecoderLayer(config, layer_idx, fast_forward=True, use_audio_attention=False)
                    )
                    layer_idx += 1
            self.layers = nn.ModuleList(layers)
        elif config.audio_adapter_type == "stack":
            self.layers = nn.ModuleList(
                [
                    LlamaDecoderLayer(config.text_config, layer_idx)
                    for layer_idx in range(config.text_config.num_hidden_layers)
                ]
            )
            layer_idx = config.text_config.num_hidden_layers
        else:
            raise NotImplementedError(f"Audio adapter type {config.audio_adapter_type} not implemented.")

        self.num_activation_checkpointing_layers = len(self.layers)

        self.decode_graph_runners = defaultdict(dict[bool, CUDAGraphRunner])
        self.norm = LlamaRMSNorm(config.text_config.hidden_size, eps=config.text_config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config.text_config)

        if not config.skip_audio_tower:
            self.audio_tower = HiggsAudioEncoder(config.audio_encoder_config)
            # self.audio_encoder_proj = HiggsAudioFeatureProjector(config)
            self.audio_encoder_proj = AudioAligner()
        else:
            self.audio_tower = None
            self.audio_encoder_proj = None
        self.audio_decoder_proj = HiggsAudioDecoderProjector(config, layer_idx=layer_idx)
        self.audio_codebook_size = (
            config.audio_codebook_size + 2
        )  # We add 1 for the audio_stream_bos token and 1 for the audio_stream_eos token

        if config.use_audio_out_embed_projector:
            self.audio_out_embed_projector = nn.Linear(
                config.text_config.hidden_size, config.text_config.hidden_size, bias=False
            )

        self.audio_codebook_embeddings = nn.Embedding(
            config.audio_num_codebooks * self.audio_codebook_size, config.text_config.hidden_size
        )

        self.audio_codebook_weights = (
            torch.ones(config.audio_num_codebooks) / config.audio_num_codebooks
        )  # default to equal weights
        self.post_init()

    def set_num_activation_checkpointing_layers(self, num_layers):
        self.num_activation_checkpointing_layers = num_layers

    def set_delay_pattern(self):
        self.config.use_delay_pattern = True
        self.use_delay_pattern = True

    def set_audio_special_tokens(self, tokenizer: AutoTokenizer):
        self.audio_out_bos_token_id = tokenizer.convert_tokens_to_ids("<|audio_out_bos|>")
        self.audio_eos_token_id = tokenizer.convert_tokens_to_ids("<|audio_eos|>")

    def _embed_audio_ids(self, audio_ids):
        """Embed the audio ids

        Args:
            audio_ids: torch.LongTensor of shape (num_codebooks, audio_in_total_length)

        Returns:
            audio_embed: torch.LongTensor of shape (audio_in_total_length, hidden_size)
        """
        codebook_shift = (
            torch.arange(self.config.audio_num_codebooks, device=audio_ids.device) * self.audio_codebook_size
        )
        audio_embed = self.audio_codebook_embeddings(audio_ids + codebook_shift.unsqueeze(-1))
        if self.config.audio_embed_avg:
            audio_embed = torch.mean(audio_embed, dim=0)
        else:
            audio_embed = torch.sum(audio_embed, dim=0)
        if self.use_audio_out_embed_projector:
            audio_embed = self.audio_out_embed_projector(audio_embed)
        return audio_embed

    def _apply_audio_tower(self, audio_features, audio_feature_attention_mask):
        """Apply the audio tower to the audio features"""

        # for param_name, param in self.audio_tower.named_parameters():
        #     print(param_name, param, 'after init')

        if audio_features.shape[0] == 0:
            if torch.is_grad_enabled():
                # FIXME!!!!!!!!
                # This is a hack to ensure that the forward+backward pass of audio_tower and audio_encoder_proj get triggered.
                # The monkey patch won't overwrite the backward pass of nn.Module.
                audio_outputs = _whisper_encoder_zero_shape_forward(
                    self.audio_tower, audio_features, attention_mask=None, check_seq_length=False
                )
                selected_audio_feature = audio_outputs.last_hidden_state
                audio_features_embed = self.audio_encoder_proj(selected_audio_feature)
                audio_feat_out_lengths = None
                return audio_features_embed, audio_feat_out_lengths
            else:
                return None, None

        audio_feat_lengths, audio_feat_out_lengths = self.audio_tower._get_feat_extract_output_lengths(
            audio_feature_attention_mask.sum(-1)
        )
        batch_size, _, max_mel_seq_len = audio_features.shape
        max_seq_len = (max_mel_seq_len - 1) // 2 + 1
        # Create a sequence tensor of shape (batch_size, max_seq_len)
        seq_range = (
            torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
            .unsqueeze(0)
            .expand(batch_size, max_seq_len)
        )
        lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
        # Create mask
        padding_mask = seq_range < lengths_expand

        if self.config._attn_implementation != "flash_attention_2":
            audio_attention_mask = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
                batch_size, 1, max_seq_len, max_seq_len
            )
        else:
            audio_attention_mask = padding_mask

        audio_outputs = self.audio_tower(audio_features, attention_mask=audio_attention_mask)
        selected_audio_feature = audio_outputs.last_hidden_state
        audio_features_embed = self.audio_encoder_proj(selected_audio_feature)

        return audio_features_embed, audio_feat_out_lengths

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    def _prepare_all_static_kv_cache_masks(self, hidden_states, attention_mask, audio_out_mask, past_key_values):
        target_length = hidden_states.shape[1]
        cur_pos = audio_out_mask.shape[1]
        min_dtype = torch.finfo(hidden_states.dtype).min
        assert len(attention_mask.shape) == 4, "Only support SDPA for now"
        kv_cache_len = past_key_values.get_max_cache_shape()
        audio_out_mask_padded = torch.nn.functional.pad(audio_out_mask, (0, kv_cache_len - cur_pos), value=True)
        fast_forward_attention_mask = attention_mask.masked_fill(
            audio_out_mask_padded[:, audio_out_mask.shape[1] - target_length : audio_out_mask.shape[1]].reshape(
                audio_out_mask_padded.shape[0], 1, target_length, 1
            )
            | audio_out_mask_padded.reshape(audio_out_mask_padded.shape[0], 1, 1, audio_out_mask_padded.shape[1]),
            min_dtype,
        )

        no_audio_out_mask = ~audio_out_mask
        no_audio_out_mask = torch.nn.functional.pad(
            no_audio_out_mask, (0, kv_cache_len - audio_out_mask.shape[1]), value=False
        )
        no_audio_out_mask = no_audio_out_mask[
            :, audio_out_mask.shape[1] - target_length : audio_out_mask.shape[1]
        ].reshape(audio_out_mask.shape[0], 1, target_length, 1) | no_audio_out_mask.reshape(
            audio_out_mask.shape[0], 1, 1, kv_cache_len
        )
        audio_attention_mask = attention_mask.masked_fill(no_audio_out_mask, min_dtype)
        return fast_forward_attention_mask, audio_attention_mask

    def _forward_core(
        self,
        hidden_states: torch.Tensor,
        text_attention_mask: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        audio_discrete_codes_mask: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]],
        use_cache: bool,
        # audio_attention_mask: torch.Tensor,
        fast_forward_attention_mask: torch.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        is_decoding_audio_token: Optional[bool] = None,
        is_using_cuda_graph: Optional[bool] = False,
    ):
        # create position embeddings to be shared across the decoder layers
        # When past_key_values is passed in, we need to offset the position ids when calculating the position embeddings.
        # Therefore, cache_position is used.
        position_id_offset = cache_position[0] if use_cache else 0
        position_embeddings = self.rotary_emb(hidden_states, position_ids + position_id_offset)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if isinstance(decoder_layer, HiggsAudioDualFFNDecoderLayer):
                layer_outputs = decoder_layer(
                    hidden_states,
                    text_attention_mask=text_attention_mask,
                    audio_attention_mask=audio_attention_mask,
                    fast_forward_attention_mask=fast_forward_attention_mask,
                    position_ids=position_ids,
                    audio_out_mask=audio_discrete_codes_mask,
                    is_decoding_audio_token=is_decoding_audio_token,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    is_using_cuda_graph=is_using_cuda_graph,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=text_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        return hidden_states, all_hidden_states, all_self_attns


    def compute_losses(self, logits, audio_logits, labels, label_audio_ids, audio_out_mask):
        """
        计算文本和音频的loss
        
        Args:
            logits: 文本logits, shape [batch_size, seq_len, vocab_size]
            audio_logits: 音频logits, shape [num_audio_tokens, num_codebooks, codebook_size] 
            labels: 文本标签, shape [batch_size, seq_len]
            label_audio_ids: 音频标签, shape [num_codebooks, audio_seq_len]
            audio_out_mask: 音频输出mask, shape [batch_size, seq_len]
        
        Returns:
            loss: 总loss
            llm_loss: 文本loss
            audio_loss: 音频loss
        """
        # 强制使用CUDA设备（如果可用）
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        
        # 初始化loss
        loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
        llm_loss = None
        audio_loss = None
        
        # 验证输入tensor并移动到目标设备
        # ("Before device transfer:")
        # i# printf logits is not None:
        #     print(f"logits device: {logits.device}, shape: {logits.shape}")
        #     logits = logits.to(device)
        # if labels is not None:
        #     print(f"labels device: {labels.device}, shape: {labels.shape}")
        #     labels = labels.to(device)
        # if audio_logits is not None:
        #     print(f"audio_logits device: {audio_logits.device}, shape: {audio_logits.shape}")
        #     audio_logits = audio_logits.to(device)
        # if label_audio_ids is not None:
        #     print(f"label_audio_ids device: {label_audio_ids.device}, shape: {label_audio_ids.shape}")
        #     label_audio_ids = label_audio_ids.to(device)
        # if audio_out_mask is not None:
        #     audio_out_mask = audio_out_mask.to(device)
        
        # print("After device transfer:")
        # if logits is not None:
        #     print(f"logits device: {logits.device}")
        # if labels is not None:
        #     print(f"labels device: {labels.device}")
        # if audio_logits is not None:
        #     print(f"audio_logits device: {audio_logits.device}")
        # if label_audio_ids is not None:
        #     print(f"label_audio_ids device: {label_audio_ids.device}")
        
        # 1. 计算文本loss (LLM loss)
        if labels is not None and logits is not None and logits.numel() > 0 and labels.numel() > 0:
            try:
                # 标准的causal language modeling loss
                shift_logits = logits[..., :-1, :].contiguous()  # [batch, seq_len-1, vocab_size]
                shift_labels = labels[..., 1:].contiguous()      # [batch, seq_len-1]
                
                # 展平
                flat_logits = shift_logits.view(-1, shift_logits.size(-1))  # [batch*(seq_len-1), vocab_size]
                flat_labels = shift_labels.view(-1)                         # [batch*(seq_len-1)]
                
                # 只计算非-100位置的loss
                # valid_mask = flat_labels != -100
                valid_mask = (flat_labels != -100) & (flat_labels != 128014)


                if valid_mask.sum() > 0:
                    llm_loss = F.cross_entropy(
                        flat_logits[valid_mask], 
                        flat_labels[valid_mask], 
                        reduction='mean'
                    )
                    loss = loss + llm_loss

                    # cur_device = flat_labels[valid_mask].device
                    # if str(cur_device) == 'cuda:7':
                    #     for i in range(1, flat_labels[valid_mask].shape[-1]):
                    #         cur_loss = F.cross_entropy(
                    #             flat_logits[valid_mask][i], 
                    #             flat_labels[valid_mask][i], 
                    #         )
                    #         if cur_loss > 1.0:
                    #             print(flat_labels[valid_mask][i-1], flat_labels[valid_mask][i], cur_loss, 'cur_loss')

                    # print(f"LLM loss computed: {llm_loss.item()}, device: {llm_loss.device}")

                    # valid_count = valid_mask.sum().item() 
                    # text_preds = torch.argmax(flat_logits[valid_mask], dim=1)  # [valid_count]
                    # print(text_preds, 'text_preds')
                    # print(flat_labels[valid_mask], 'flat_labels')
                    # correct_count = (text_preds == flat_labels[valid_mask]).sum().item()
                    # text_acc = correct_count / valid_count  # 浮点数（0~1）
                    # print(f"Text accuracy: {text_acc} total: {valid_count}")
                    # print('\n')
                else:
                    llm_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
                    print("No valid LLM tokens to compute loss")
            except Exception as e:
                print(f"Error computing LLM loss: {e}")
                llm_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
        else:
            llm_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
            # print("Skipping LLM loss computation - invalid inputs")
        
        # 2. 计算音频loss
        if (audio_logits is not None and 
            label_audio_ids is not None and 
            audio_logits.numel() > 0 and 
            label_audio_ids.numel() > 0
            ):
            
            try:
                # audio_logits: [num_audio_tokens, num_codebooks, codebook_size]
                # label_audio_ids: [num_codebooks, audio_seq_len]
                
                num_audio_tokens = audio_logits.shape[0]
                num_codebooks = min(audio_logits.shape[1], label_audio_ids.shape[0])
                
                # print(f"Audio loss computation: num_audio_tokens={num_audio_tokens}, num_codebooks={num_codebooks}")
                
                if num_audio_tokens > 0 and num_codebooks > 0:
                    audio_losses = []
                    
                    # 使用label_audio_ids作为音频标签
                    assert num_audio_tokens == label_audio_ids.shape[1]
                    audio_seq_len = min(num_audio_tokens, label_audio_ids.shape[1])

                    # print('\n')
                    # print(num_audio_tokens, 'num_audio_tokens', label_audio_ids.shape[1], 'label_audio_ids.shape[1]', audio_seq_len, 'audio_seq_len')
                    # print(audio_logits.shape, 'audio_logits.shape')
                    # print(label_audio_ids.shape, label_audio_ids, 'label_audio_ids')
                    # print('\n')
                    
                    for cb in range(num_codebooks):
                        cb_logits = audio_logits[:audio_seq_len, cb, :]  # [audio_seq_len, codebook_size]
                        cb_labels = label_audio_ids[cb, :audio_seq_len]  # [audio_seq_len]
                        
                        valid_mask = cb_labels != -100
                        valid_count = valid_mask.sum().item() 

                        # 过滤掉-100的标签
                        # valid_mask = cb_labels != -100
                        if valid_mask.sum() > 0:
                            cb_loss = F.cross_entropy(
                                cb_logits[valid_mask], 
                                cb_labels[valid_mask], 
                                reduction='mean'
                            )
                            audio_losses.append(cb_loss)
                            # print(f"Codebook {cb} loss: {cb_loss.item()}")

                            cb_preds = torch.argmax(cb_logits[valid_mask], dim=1)  # [valid_count]
                            correct_count = (cb_preds == cb_labels[valid_mask]).sum().item()
                            cb_acc = correct_count / valid_count  # 浮点数（0~1）
                            # print(f"Codebook {cb} accuracy: {cb_acc} total: {valid_count}")
                    # edit audio_losses
                    # audio_losses = None
                    if audio_losses:
                        audio_loss = torch.stack(audio_losses).mean()
                        loss = loss + audio_loss
                        # print(f"Audio loss computed: {audio_loss.item()}, device: {audio_loss.device}")
                    else:
                        audio_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
                        print("No valid audio tokens to compute loss")
                else:
                    audio_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
                    print("No audio tokens or codebooks available")
                    
            except Exception as e:
                print(f"Error computing audio loss: {e}")
                audio_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
        else:
            audio_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
            # print("Skipping audio loss computation - invalid inputs or no audio output")
        
        # print(f"Final loss: {loss.item()}, device: {loss.device}")
        # print(f"LLM loss: {llm_loss.item() if llm_loss is not None else 'None'}")
        # print(f"Audio loss: {audio_loss.item() if audio_loss is not None else 'None'}")
        
        return loss, llm_loss, audio_loss
    

    # def compute_losses(self, logits, audio_logits, labels, label_audio_ids, audio_out_mask):
        """
        计算文本和音频的loss (V2 - 修正版)
        
        Args:
            logits: 文本logits, shape [batch_size, seq_len, vocab_size]
            audio_logits: 音频logits, shape [num_audio_tokens, num_codebooks, codebook_size] 
            labels: 文本标签, shape [batch_size, seq_len]
            label_audio_ids: 音频标签, shape [num_codebooks, num_audio_tokens_in_batch]
            audio_out_mask: 音频输出mask, shape [batch_size, seq_len]
        
        Returns:
            loss: 总loss
            llm_loss: 文本loss
            audio_loss: 音频loss
        """
        device = logits.device if logits is not None else (audio_logits.device if audio_logits is not None else 'cpu')
        
        # --- 初始化loss值为 requires_grad=True 的张量，确保即使某个loss为0也能参与计算图 ---
        llm_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
        audio_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)

        # 1. 计算文本loss (LLM loss) 
        if labels is not None and logits is not None and logits.numel() > 0 and labels.numel() > 0:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)

            # chosen_token = torch.argmax(flat_logits, dim=-1)
            # print('chosen_token', chosen_token)
            # print('flat_labels ', flat_labels)
            
            # 只计算非-100位置的loss (HuggingFace标准做法)
            valid_mask = flat_labels != -100
            if valid_mask.sum() > 0:
                llm_loss = F.cross_entropy(
                    flat_logits[valid_mask], 
                    flat_labels[valid_mask], 
                    reduction='mean'
                )

        # 2. 计算音频loss (Audio loss) - *** 这是关键的修正部分 ***
        if (audio_logits is not None and 
            label_audio_ids is not None and 
            audio_logits.numel() > 0 and 
            label_audio_ids.numel() > 0):

            # --- 断言：检查进入loss计算前，logits和labels的序列长度是否一致 ---
            # audio_logits 的第一维是序列长度，即整个batch中audio token的总数
            num_audio_tokens_from_logits = audio_logits.shape[0]
            # label_audio_ids 的第二维是序列长度
            num_audio_tokens_from_labels = label_audio_ids.shape[1]

            # 这个断言至关重要，如果它失败，说明数据整理(collator)或模型forward逻辑有问题
            assert num_audio_tokens_from_logits == num_audio_tokens_from_labels, (
                f"音频Logits和Labels的序列长度不匹配! "
                f"Logits长度: {num_audio_tokens_from_logits}, "
                f"Labels长度: {num_audio_tokens_from_labels}. "
                "请检查数据整理(Collator)和HiggsAudioDataset的实现。"
            )

            # --- 核心修正：正确的自回归错位学习 ---
            # 目标：用第t个token的预测(logits)去和第t+1个token的真值(label)做比较
            # 因此，我们需要对齐 audio_logits 和 label_audio_ids，并都去掉序列中的一个元素

            # 使用 0 到 n-1 时刻的 audio_logits
            shifted_logits = audio_logits[:-1, :, :].contiguous()
            # 使用 1 到 n 时刻的 label_audio_ids
            shifted_labels = label_audio_ids[:, 1:].contiguous()

            # 再次断言，确保错位后的长度仍然一致
            assert shifted_logits.shape[0] == shifted_labels.shape[1], (
                "错位后的音频Logits和Labels长度不匹配!"
            )
            
            # 如果序列长度大于1，才有可能计算loss
            if shifted_logits.shape[0] > 0:
                num_codebooks = min(shifted_logits.shape[1], shifted_labels.shape[0])
                audio_losses = []

                for cb in range(num_codebooks):
                    # cb_logits shape: [seq_len-1, codebook_size]
                    cb_logits = shifted_logits[:, cb, :]
                    # cb_labels shape: [seq_len-1]
                    cb_labels = shifted_labels[cb, :]

                    cb_predict = torch.argmax(cb_logits, dim=1)
                    cb_accuracy = (cb_predict == cb_labels).sum() / cb_predict.shape[0]
                    
                    # 同样，只对非-100的标签计算loss
                    valid_mask = cb_labels != -100
                    if valid_mask.sum() > 0:
                        cb_loss = F.cross_entropy(
                            cb_logits[valid_mask], 
                            cb_labels[valid_mask], 
                            reduction='mean'
                        )
                        audio_losses.append(cb_loss)
                        # print(cb, '-th codebook', cb_accuracy, 'total', cb_predict.shape[0], 'loss', cb_loss.item())
                
                if audio_losses:
                    # 对所有codebook的loss求平均
                    audio_loss = torch.stack(audio_losses).mean()

        # --- 合并总Loss ---
        # 只有当loss大于0时才相加，避免不必要的计算
        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
        if llm_loss.item() > 0:
            total_loss = total_loss + llm_loss
        if audio_loss.item() > 0:
            total_loss = total_loss + audio_loss
            
        return total_loss, llm_loss, audio_loss


    def shift_tokens_right(self, input_ids: torch.Tensor, fill_token_id: int, start_ids: torch.Tensor):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, :-1] = input_ids[:, 1:].clone()
        # shifted_input_ids[:, -1] = fill_token_id

        shifted_input_ids[:, start_ids - 1] = fill_token_id

        return shifted_input_ids

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_feature_attention_mask: Optional[torch.BoolTensor] = None,
        audio_in_ids: Optional[torch.LongTensor] = None,
        audio_in_ids_start: Optional[torch.LongTensor] = None,
        audio_out_ids: Optional[torch.LongTensor] = None,
        audio_out_ids_start: Optional[torch.LongTensor] = None,
        audio_out_ids_start_group_loc: Optional[torch.LongTensor] = None,
        label_ids: Optional[torch.LongTensor] = None,
        label_audio_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_audio_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cache_audio_discrete_codes_mask: Optional[torch.LongTensor] = None,
        past_key_values_buckets: Optional[OrderedDict[int, Cache]] = None,
        reward: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        all_input_ids_for_infer: Optional[torch.LongTensor] = None,
    ):
        """Forward pass for the Higgs-Audio model.

        Args:
            input_ids (:obj:`torch.LongTensor`):
                The input ids of the prompt. It will have shape (bsz, seq_len).
                When use_cache is enabled, the input_ids will have
                shape (bsz, 1) for incremental decode or None
            inputs_embeds:
                Input embeddings. This flag won't be used.
            attention_mask (:obj:`torch.LongTensor`):
                The attention mask of the prompt. It will have shape (bsz, seq_len).
            audio_features (:obj:`torch.FloatTensor`):
                The audio features extracted by Whisper. It will have shape (num_audio_in, feature_dim, max_mel_seq_len).
            audio_feature_attention_mask (:obj:`torch.LongTensor`):
                The attention mask of the audio features. It will have shape (num_audio_in, max_mel_seq_len).
            audio_in_ids (:obj:`torch.LongTensor`):
                The discretized audio tokens. It will have shape (num_codebooks, audio_in_total_length).
            audio_in_ids_start (:obj:`torch.LongTensor`):
                The start indices for each audio in audio_in_ids. It will have shape (num_audio_in,)
            audio_out_ids (:obj:`torch.LongTensor`):
                The discretized audio tokens. It will have shape (num_codebooks, audio_out_total_length).
            audio_out_ids_start (:obj:`torch.LongTensor`):
                The start indices for each audio in audio_out_ids. It will have shape (num_audio_out,)
            audio_out_ids_start_group_loc (:obj:`torch.LongTensor`):
                The sample indices in a batch that map to each element in the audio_out_ids_start. It will have shape (num_audio_out,)
            label_text_ids (:obj:`torch.LongTensor`):
                The labels of the prompt. It will have shape (bsz, seq_len).
            label_audio_ids (:obj:`torch.LongTensor`):
                The labels of the audio tokens. It will have the same shape as audio_out_ids, i.e., (num_codebooks, audio_out_total_length)
            past_key_values (:obj:`Tuple`):
                Tuple of past key values.
            use_cache (:obj:`bool`):
                Whether to use cache.
            output_attentions (:obj:`bool`):
                Whether to output attentions.
            output_hidden_states (:obj:`bool`):
                Whether to output hidden states.
            output_audio_hidden_states (:obj:`bool`):
                Whether to output audio hidden states.
            return_dict (:obj:`bool`):
                Whether to return a dictionary.
            cache_position (:obj:`torch.LongTensor`):
                The position of the cache.
            cache_audio_discrete_codes_mask (:obj:`torch.LongTensor`):
                The cached audio discrete codes mask. It will only be used when use_cache is turned on.
            past_key_values_buckets (:obj:`OrderedDict`):
                The buckets of past key values.
        """
        with nvtx.range("test_prepare"):
            target_device = input_ids.device
            label_audio_ids_save = label_audio_ids
            # not used
            del inputs_embeds

            if audio_features is not None:
                audio_features = audio_features.to(target_device)
                audio_feature_attention_mask = audio_feature_attention_mask.to(target_device)

            # 1. Extract the input embeddings
            inputs_embeds = self.embed_tokens(input_ids)

            # 2. Extract audio embeddings
            if self.config.skip_audio_tower:
                audio_features_embed = audio_features_length = None
            else:
                with nvtx.range("test_whisper"):
                    audio_features_embed, audio_features_length = self._apply_audio_tower(
                        audio_features, audio_feature_attention_mask
                    )

            if self.config.encode_audio_in_tokens:
                if audio_in_ids is not None and audio_in_ids.shape[-1] > 0:
                    audio_in_ids = audio_in_ids.to(target_device)
                else:
                    audio_in_ids = torch.zeros((self.audio_num_codebooks, 0), device=target_device, dtype=torch.long)
                audio_in_embed = self._embed_audio_ids(audio_in_ids).bfloat16()
            else:
                audio_in_embed = None

            if audio_out_ids is not None and audio_out_ids.shape[-1] > 0:
                audio_out_ids = audio_out_ids.to(target_device)
            else:
                audio_out_ids = torch.zeros((self.audio_num_codebooks, 0), device=target_device, dtype=torch.long)
            audio_out_embed = self._embed_audio_ids(audio_out_ids)

        # 3. Merge text, audio-in embeddings, and audio-out embeddings

        # use_cache is turned on during inference time, we should set round_to to 1 to avoid extra padding in the end.

        # print('\n')
        # print('input_ids before merge', input_ids, input_ids.shape, '\n')
        # print('label_ids before merge', label_ids, label_ids.shape, '\n\n\n\n')
        # # # print(audio_features_embed.shape, 'audio_features_embed.shape')
        # # # print(audio_features_length, 'audio_features_length')
        # print(audio_in_embed.shape, 'audio_in_embed.shape')
        # print(audio_in_ids_start, 'audio_in_ids_start')
        # print(audio_out_embed.shape, 'audio_out_embed.shape')
        # print(audio_out_ids_start, 'audio_out_ids_start')

        # print(inputs_embeds.shape, 'inputs_embeds.shape before merge')
        # print('='*20)
        with nvtx.range("test_merge"):
            round_to = 1 if use_cache else 8
            left_padding = True if use_cache or input_ids.shape[0] == 1 else False
            (
                inputs_embeds,
                attention_mask,
                labels,
                position_ids,
                input_ids,
                audio_in_mask,
                audio_in_discrete_codes_mask,
                audio_out_mask,
            # ) = merge_input_ids_with_audio_features(
            ) = merge_input_ids_with_audio_features_without_audio_in_embed(
                audio_features_embed,
                audio_features_length,
                audio_in_embed,
                audio_in_ids_start,
                audio_out_embed,
                audio_out_ids_start,
                self.audio_in_token_idx,
                self.audio_out_token_idx,
                inputs_embeds,
                input_ids,
                attention_mask,
                label_ids,
                pad_token_id=self.padding_idx,
                round_to=round_to,
                left_padding=left_padding,
            )

        # print(inputs_embeds.shape, 'inputs_embeds.shape after merge')
        # print(input_ids, 'input_ids after merge', input_ids.shape, '\n')
        # print(labels, 'labels after merge', labels.shape, '\n\n\n\n')
        # print(audio_in_mask, 'audio_in_mask', audio_in_mask.shape)
        # print(audio_in_discrete_codes_mask.sum().item(), 'audio_in_discrete_codes_mask', audio_in_discrete_codes_mask.shape)
        # print(audio_out_mask, 'audio_out_mask', audio_out_mask.shape)
        # print('\n')

        # print('\n')
        # print(audio_in_ids)
        # print(inputs_embeds[audio_in_mask], 'audio_in_embed in inputs_embeds')
        # print(audio_in_embed, 'audio_in_embed')
        # print(audio_out_ids)
        # print(inputs_embeds[audio_out_mask], 'audio_out_embed in inputs_embeds')
        # print(audio_out_embed, 'audio_out_embed')
        # print('\n')

        # re-check if we use the correct kv cache bucket after
        # the input_embeds has been merged with audio features
        if past_key_values_buckets is not None and inputs_embeds.shape[1] > past_key_values.get_max_cache_shape():
            past_key_values, self.current_past_key_values_bucket = self._prepare_kv_cache(
                inputs_embeds.shape[1], None, past_key_values_buckets
            )

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
            if isinstance(past_key_values, StaticCache) and past_seen_tokens >= past_key_values.get_max_cache_shape():
                raise ValueError(
                    f"The current sequence length ({past_seen_tokens}) exceeds "
                    f"the maximum cache shape. "
                    f"Please consider increasing the cache size."
                )

        # Use torch compile
        use_static_cache = isinstance(past_key_values, StaticCache)

        # Apply the LLM component
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        audio_discrete_codes_mask = audio_in_discrete_codes_mask | audio_out_mask
        if cache_audio_discrete_codes_mask is not None and use_cache:
            audio_discrete_codes_mask = torch.concat(
                [cache_audio_discrete_codes_mask, audio_discrete_codes_mask], dim=1
            )

        # Generate the audio attention mask outside the layer to avoid recompilation
        if use_static_cache:
            fast_forward_attention_mask, audio_attention_mask = self._prepare_all_static_kv_cache_masks(
                hidden_states, causal_mask, audio_discrete_codes_mask, past_key_values
            )
            # Set the audio out mask to the last token
            if hidden_states.shape[1] == 1:
                audio_discrete_codes_mask = audio_discrete_codes_mask[:, -1:]
                audio_discrete_codes_mask = audio_discrete_codes_mask.reshape((-1, 1)).contiguous()
                is_decoding_audio_token = audio_discrete_codes_mask.item()
            else:
                is_decoding_audio_token = False

        # Use the captured cuda graph runner for decoding
        # if it exists, otherwise use the normal forward pass
        if (
            past_key_values is not None
            and past_key_values.get_max_cache_shape() in self.decode_graph_runners
            and (input_ids.shape[-1] == 1)
        ):
            _forward_core = self.decode_graph_runners[past_key_values.get_max_cache_shape()][is_decoding_audio_token]
            is_using_cuda_graph = True
        else:
            _forward_core = self._forward_core
            is_using_cuda_graph = False

        # time_before_get_complex = time.time()

        with nvtx.range("test_attention_mask"):
            has_audio_out = audio_out_mask is not None and audio_out_mask.shape[0] > 0
            if has_audio_out:
                # print(audio_out_mask, 'audio_out_mask here!!!\n')
                # causal_mask = create_interleaved_modal_mask(
                #     audio_out_mask=audio_out_mask,
                #     hidden_states=inputs_embeds,
                #     attention_mask=causal_mask,  # 4D因果掩码或None
                #     cache_position=cache_position,
                # )

                """
                audio_out_mask: torch.BoolTensor,  # 原有：A位置=True，其他=False
                input_ids: torch.LongTensor,       # 新增：输入序列的token ID，用于识别<audio_out_bos>/<audio_eos>
                audio_special_ids: Tuple[int, int],# 新增：(audio_out_bos_id, audio_eos_id)，固定标记的ID
                """
                # 1. 定义音频相关标记的固定ID（需替换为你实际的token ID）
                audio_out_bos_id = 128013
                audio_out_eos_id = 128014
                audio_out_token_idx = 128016
                audio_out_last_bos_id = 128030
                audio_special_ids = (audio_out_bos_id, audio_out_last_bos_id, audio_out_eos_id)
                if past_key_values is None:
                    causal_mask, audio_causal_mask = create_interleaved_modal_mask_complex(
                        audio_out_mask=audio_out_mask,
                        input_ids=input_ids,  # 模型输入的token ID序列（B, seq_len）
                        audio_special_ids=audio_special_ids,
                        hidden_states=inputs_embeds,
                        attention_mask=causal_mask,  # 4D因果掩码或None
                        cache_position=None,
                    )
                else:
                    if input_ids.shape[-1] > 5:
                        self.all_input_ids_for_infer = input_ids
                    causal_mask, audio_causal_mask = create_interleaved_modal_mask_complex(
                        audio_out_mask=audio_out_mask,
                        input_ids=self.all_input_ids_for_infer,  # 模型输入的token ID序列（B, seq_len）
                        audio_special_ids=audio_special_ids,
                        hidden_states=inputs_embeds,
                        attention_mask=causal_mask,  # 4D因果掩码或None
                        cache_position=cache_position,
                    )

        # time_after_get_complex = time.time()
        # print(f"Time get complex: {time_after_get_complex - time_before_get_complex} {str(causal_mask.device)}\n\n\n")

        # if causal_mask is not None:
        #     cur_device = causal_mask.device
        #     # print(cur_device,'\n')
        #     if str(cur_device) == 'cuda:1':
        #         print(audio_out_mask.shape, audio_out_mask, 'audio_out_mask\n')
        #         print(input_ids.shape, input_ids, 'input_ids\n')
        #         print(position_ids.shape, position_ids, 'position_ids\n')
        #         print(causal_mask.shape, causal_mask, 'causal_mask\n')
        #         print(audio_causal_mask.shape, audio_causal_mask, 'audio_causal_mask\n')
        #         print(cache_position.shape, cache_position, 'cache_position\n')
        #         token_num = causal_mask.shape[3]
        #         for i in range(causal_mask.shape[2]):
        #             for j in range(causal_mask.shape[3]):
        #                 if causal_mask[0][0][i][j] < -1e5 and j <= i:
        #                     print(i, j, input_ids[0][i], input_ids[0][j], 'mask place\n')
        #         for i in range(causal_mask.shape[2]):
        #             for j in range(causal_mask.shape[3]):
        #                 if causal_mask[0][0][i][j] > -1e5 and j > i:
        #                     print(i, j, input_ids[0][i], input_ids[0][j], 'text causal_mask error\n')

        #         for i in range(audio_causal_mask.shape[2]):
        #             for j in range(audio_causal_mask.shape[3]):
        #                 if audio_causal_mask[0][0][i][j] < -1e5 and j <= i:
        #                     print(i, j, input_ids[0][i], input_ids[0][j], 'mask_audio place\n')
        #         for i in range(audio_causal_mask.shape[2]):
        #             for j in range(audio_causal_mask.shape[3]):
        #                 if audio_causal_mask[0][0][i][j] > -1e5 and j > i:
        #                     print(i, j, input_ids[0][i], input_ids[0][j], 'audio_causal_mask error\n')

        #     import time
        #     time.sleep(10000)

        # debug_device = str(causal_mask.device)
        # if debug_device == 'cuda:0':
        #     time_before_forward_core = time.time()

        with nvtx.range("test_forward_core"):
            hidden_states, all_hidden_states, all_self_attns = _forward_core(
                hidden_states=hidden_states,
                text_attention_mask=causal_mask,
                audio_attention_mask=audio_causal_mask,
                position_ids=position_ids,
                audio_discrete_codes_mask=audio_discrete_codes_mask,
                is_decoding_audio_token=is_decoding_audio_token if use_static_cache else None,
                cache_position=cache_position,
                past_key_values=past_key_values,
                use_cache=use_cache,
                # audio_attention_mask=audio_attention_mask if use_static_cache else None,
                fast_forward_attention_mask=fast_forward_attention_mask if use_static_cache else None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                is_using_cuda_graph=is_using_cuda_graph,
            )
            hidden_states = self.norm(hidden_states)

        # if debug_device == 'cuda:0':
        #     time_after_forward_core = time.time()
        #     print(f"Time forward core: {time_after_forward_core - time_before_forward_core}\n")

        # # add hidden states from the last decoder layer
        # if output_hidden_states:
        #     all_hidden_states += (hidden_states,)
        # Apply the audio decoder projector
        # logits, audio_logits, decoder_all_self_attns, decoder_all_hidden_states, audio_hidden_states, _ = (
        #     self.audio_decoder_proj(
        #         hidden_states,
        #         audio_out_mask,
        #         label_audio_ids=label_audio_ids,
        #         attention_mask=causal_mask,
        #         position_ids=position_ids,
        #         past_key_values=past_key_values,
        #         use_cache=use_cache,
        #         output_attentions=output_attentions,
        #         output_audio_hidden_states=output_audio_hidden_states,
        #         cache_position=cache_position,
        #     )
        # )

        logits, audio_logits, decoder_all_self_attns, decoder_all_hidden_states, audio_hidden_states, _ = (
            self.audio_decoder_proj(
                hidden_states,
                audio_out_mask,
            )
        )

        # if debug_device == 'cuda:0':
        #     time_after_decode_proj = time.time()
        #     print(f"Time decode_proj: {time_after_decode_proj - time_after_forward_core}\n\n\n")

        if audio_logits is not None:
            audio_logits = audio_logits.view(
                audio_logits.shape[0], self.audio_num_codebooks, self.audio_codebook_size
            ).float()

        if output_hidden_states:
            if decoder_all_hidden_states is not None and len(decoder_all_hidden_states) > 1:
                all_hidden_states += decoder_all_hidden_states[1:]

        if output_attentions:
            all_self_attns += decoder_all_self_attns

        next_cache = past_key_values if use_cache else None

        if label_audio_ids is not None:
            label_audio_ids = self.shift_tokens_right(label_audio_ids, -100, audio_out_ids_start)

        # if debug_device == 'cuda:0':
        #     time_before_compute_loss = time.time()
        #     print(f"Time shift_tokens_right: {time_before_compute_loss - time_after_decode_proj}\n")

        if past_key_values is None:
            loss, llm_loss, audio_loss = self.compute_losses(
                logits, audio_logits, labels, label_audio_ids, audio_out_mask
            )

            if audio_loss is not None and audio_loss > 1e-5:
                print(llm_loss, 'llm_loss', audio_loss, 'audio_loss')
        else:
            loss, llm_loss, audio_loss = 0.0, 0.0, 0.0

        # if debug_device == 'cuda:0':
        #     time_after_compute_loss = time.time()
        #     print(f"Time compute_losses: {time_after_compute_loss - time_before_compute_loss}\n")

        ret = HiggsAudioModelOutputWithPast(
            loss=loss,
            llm_loss=llm_loss,
            logits=logits,
            audio_logits=audio_logits,
            expanded_input_ids=input_ids,
            expanded_labels=labels,
            audio_in_mask=audio_in_mask,
            audio_in_discrete_codes_mask=audio_in_discrete_codes_mask,
            audio_out_mask=audio_out_mask,
            attention_mask=attention_mask,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            audio_hidden_states=audio_hidden_states,
            attentions=all_self_attns,
        )

        # ret = HiggsAudioModelOutputWithPast(
        #     logits=logits,
        #     audio_logits=audio_logits,
        #     expanded_input_ids=input_ids,
        #     expanded_labels=labels,
        #     audio_in_mask=audio_in_mask,
        #     audio_in_discrete_codes_mask=audio_in_discrete_codes_mask,
        #     audio_out_mask=audio_out_mask,
        #     attention_mask=attention_mask,
        #     past_key_values=next_cache,
        #     hidden_states=all_hidden_states,
        #     audio_hidden_states=audio_hidden_states,
        #     attentions=all_self_attns,
        # )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if not return_dict:
            outputs = ret.to_tuple()
            return outputs

        return ret

    # Overwrite GenerationMixin._update_model_kwargs_for_generation
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
        extend_attention_mask: bool = True,
    ) -> Dict[str, Any]:
        """Update the model kwargs for each step."""
        model_kwargs["past_key_values"] = outputs.past_key_values

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            if extend_attention_mask:
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        if "cache_audio_discrete_codes_mask" in model_kwargs:
            if model_kwargs["cache_audio_discrete_codes_mask"] is None:
                model_kwargs["cache_audio_discrete_codes_mask"] = (
                    outputs.audio_in_discrete_codes_mask | outputs.audio_out_mask
                )
            else:
                model_kwargs["cache_audio_discrete_codes_mask"] = torch.concat(
                    [
                        model_kwargs["cache_audio_discrete_codes_mask"],
                        outputs.audio_in_discrete_codes_mask | outputs.audio_out_mask,
                    ],
                    1,
                )

        return model_kwargs

    def _copy_kv_cache(self, from_cache: Cache, to_cache: Cache):
        num_layers = self.config.text_config.num_hidden_layers
        if self.config.audio_dual_ffn_layers is not None:
            num_layers += len(self.config.audio_dual_ffn_layers)
        """ Copy the key-value pairs from one cache to another. """
        for layer_idx in range(num_layers):
            from_cache_size = from_cache.get_max_cache_shape()
            assert to_cache.get_max_cache_shape() >= from_cache_size, (
                f"The target cache size {to_cache.get_max_cache_shape()} is smaller than the source cache size {from_cache_size}."
            )
            to_cache.key_cache[layer_idx][:, :, :from_cache_size, :] = from_cache.key_cache[layer_idx]
            to_cache.value_cache[layer_idx][:, :, :from_cache_size, :] = from_cache.value_cache[layer_idx]

    def _prepare_kv_cache(
        self,
        current_sequence_length: int,
        current_past_key_values_bucket: Optional[int],
        past_key_values_buckets: OrderedDict[int, Cache],
    ) -> Tuple[Optional[Cache], Optional[int]]:
        """Prepare the KV cache for the current sequence length."""
        for cache_length in past_key_values_buckets.keys():
            if cache_length >= current_sequence_length:
                # Promote to the next KV cache bucket, copy the current KV cache bucket
                # to the new one.
                if current_past_key_values_bucket is not None and cache_length != current_past_key_values_bucket:
                    self._copy_kv_cache(
                        past_key_values_buckets[current_past_key_values_bucket], past_key_values_buckets[cache_length]
                    )

                return past_key_values_buckets[cache_length], cache_length

        raise ValueError(
            f"The current sequence length {current_sequence_length} is larger than "
            f"all past key values buckets {past_key_values_buckets.keys()}."
        )

    def _sample_audio_tokens(
        self,
        hidden_states: torch.Tensor,
        audio_logits: torch.Tensor,
        audio_out_ids: torch.Tensor,
        do_sample: bool,
        logits_processor: LogitsProcessorList,
        device: torch.device,
        torch_generator: Optional[torch.Generator],
        generation_config: GenerationConfig,
        num_delay: int,
        num_remaining_delays: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, Optional[int]]:
        """Sample audio tokens and its corresponding text tokens from the logits"""

        # parameters related to repetition aware sampling
        ras_win_len = generation_config.generation_kwargs.get("ras_win_len", None)
        ras_win_max_num_repeat = generation_config.generation_kwargs.get("ras_win_max_num_repeat", 2)
        audio_eos_token_id = generation_config.generation_kwargs.get("audio_eos_token_id", None)
        audio_eos_token_id = 128014
        # In the audio generation mode, we sample from audio_logits and keep updating audio_out_ids.
        next_audio_token_logits = audio_logits.clone()[-1, :, :].float().to(device)
        # TopP, TopK logits processor supports empty input_ids
        next_audio_token_scores = logits_processor(None, next_audio_token_logits)

        # token selection
        if do_sample:
            # next_audio_token_scores has been applied top_p, top_k, and temperature.
            probs = nn.functional.softmax(next_audio_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_audio_tokens = torch.multinomial(probs, num_samples=1, generator=torch_generator).squeeze(1)
        else:
            next_audio_tokens = torch.argmax(next_audio_token_scores, dim=-1)

        # next_tokens: (num_codebooks, )
        if ras_win_len is not None:
            # check if there are repetitions over a window of tokens.
            rep_num = (audio_out_ids[:, -ras_win_len:] == next_audio_tokens.unsqueeze(1)).sum(dim=1)

            # if we saw repeated tokens in the most recent window of tokens, resample without temperature.
            row_indices = torch.nonzero(rep_num >= ras_win_max_num_repeat).squeeze(1)
            resampled_next_tokens = (
                next_audio_token_logits[row_indices]
                .softmax(dim=-1)
                .multinomial(1, replacement=True, generator=torch_generator)
                .squeeze(1)
            )
            next_audio_tokens[row_indices] = resampled_next_tokens

        # Force the next text tokens to be <|AUDIO_OUT|> in audio generation mode
        next_tokens = torch.full(
            (audio_logits.shape[0],),
            self.config.audio_out_token_idx,
            dtype=torch.long,
            device=device,
        )

        # Handle delay_pattern
        if self.use_delay_pattern:
            if num_delay + 1 < next_audio_tokens.shape[0]:
                next_audio_tokens[(num_delay + 1) :] = self.config.audio_stream_bos_id
                num_delay += 1
            if num_remaining_delays is not None:
                next_audio_tokens[: (self.audio_num_codebooks - num_remaining_delays)] = (
                    self.config.audio_stream_eos_id
                )
                num_remaining_delays -= 1
            else:
                all_eos_indices = (next_audio_tokens == self.config.audio_stream_eos_id).nonzero()
                if torch.numel(all_eos_indices) > 0:
                    all_eos_indices = all_eos_indices[0]
                    last_eos_idx = all_eos_indices[-1]
                    next_audio_tokens[:last_eos_idx] = self.config.audio_stream_eos_id
                    num_remaining_delays = self.audio_num_codebooks - last_eos_idx - 1
            if num_remaining_delays is not None and num_remaining_delays <= 0:
                next_tokens[...] = audio_eos_token_id
                num_delay = 0
                num_remaining_delays = None

        return (
            next_tokens,
            next_audio_tokens,
            next_audio_token_logits,
            next_audio_token_scores,
            num_delay,
            num_remaining_delays,
        )

    def _sample_text_tokens(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        do_sample: bool,
        logits_processor: LogitsProcessorList,
        device: torch.device,
        generation_mode: GenerationMode,
        torch_generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Sample text tokens from the logits"""
        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = logits.clone()[:, -1, :].float()
        next_token_logits = next_token_logits.to(input_ids.device)

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        if generation_mode == GenerationMode.AUDIO_INIT:
            # See the audio bos token, we should start generating audio tokens
            next_tokens = torch.full(
                (input_ids.shape[0],),
                self.audio_out_token_idx,
                dtype=torch.long,
                device=device,
            )
            next_audio_tokens = torch.full(
                (self.config.audio_num_codebooks,),
                self.config.audio_stream_bos_id,
                dtype=torch.long,
                device=device,
            )
        else:
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1, generator=torch_generator).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            next_audio_tokens = None

        return next_tokens, next_audio_tokens, next_token_logits, next_token_scores

    # Built on top of GenerationMixin._sample.
    # We revise the implementation to support generating both audio / text.
    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        past_key_values_buckets: Optional[OrderedDict[int, Cache]],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for joint text/audio models using **multinomial sampling**.

        This function may also be revised to support generating samples from HiggsAudio-like end-to-end text/audio models built on top of LLMs.
        If the input_ids ends with <|audio_out_bos|>, we will switch to the audio-generation mode.

        ```
        ...<|start_header_id|>assistant<|end_header_id|>\n\n<|audio_out_bos|>
        ```

        Otherwise, we will keep generating the text tokens.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        with nvtx.range("_sample part1:"):
            assert input_ids.shape[0] == 1, "Only support batch_size=1 in _sample()"
            audio_out_bos_token_id = generation_config.generation_kwargs.get("audio_out_bos_token_id", None)

            audio_out_last_bos_token_id = 128030

            # torch generator for sampling
            seed = generation_config.generation_kwargs.get("seed", None)
            if seed is not None:
                torch_generator = torch.Generator(device=input_ids.device).manual_seed(seed)
            else:
                torch_generator = None

            # init values
            pad_token_id = generation_config._pad_token_tensor
            output_attentions = generation_config.output_attentions
            output_hidden_states = generation_config.output_hidden_states
            output_scores = generation_config.output_scores
            output_logits = generation_config.output_logits
            output_logits = True
            return_dict_in_generate = generation_config.return_dict_in_generate
            return_dict_in_generate = True
            max_length = generation_config.max_length
            has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
            do_sample = generation_config.do_sample
            # Used to track which past_key_va
            self.current_past_key_values_bucket = None

            # init attention / hidden states / scores tuples
            scores = () if (return_dict_in_generate and output_scores) else None
            raw_logits = () if (return_dict_in_generate and output_logits) else None

            decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
            decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

            # keep track of which sequences are already finished
            batch_size, cur_len = input_ids.shape
            this_peer_finished = False
            unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
            if generation_config.use_cache:
                model_kwargs["cache_audio_discrete_codes_mask"] = None

            init_model_input = True
            num_delay = 0
            num_remaining_delays = None
            audio_sequences = []
            # A tensor to keep track of all the audio placeholder tokens.
            input_ids_full = input_ids.clone()

        with nvtx.range("_sample part2:"):
            # Initialize the audio variables based on the input prompt.
            if input_ids[0][-1] == self.config.audio_out_token_idx:
                audio_sequences = [model_kwargs["audio_out_ids"][:, model_kwargs["audio_out_ids_start"][-1] :]]
                if self.use_delay_pattern:
                    num_delay = (
                        self.audio_num_codebooks
                        - (model_kwargs["audio_out_ids"][:, -1] == self.config.audio_stream_bos_id).sum()
                    )
                    all_eos_indices = (model_kwargs["audio_out_ids"][:, -1] == self.config.audio_stream_eos_id).nonzero()
                    if torch.numel(all_eos_indices) > 0:
                        all_eos_indices = all_eos_indices[0]
                        last_eos_idx = all_eos_indices[-1]
                        num_remaining_delays = self.audio_num_codebooks - last_eos_idx - 1

        self.all_input_ids_for_infer = None

        # with nvtx.range("_sample part3:"):
        #     with nvtx.range("_sample part3.1:"):
        #         audio_features_embed, audio_features_length = self._apply_audio_tower(
        #             model_kwargs["audio_features"], model_kwargs["audio_feature_attention_mask"]
        #         )
        #     with nvtx.range("_sample part3.2:"):
        #         # print(audio_features_length, 'audio_features_length\n')
        #         with nvtx.range("_sample part3.2.1:"):
        #             add_tokens = torch.sum(audio_features_length, -1) - audio_features_length.shape[0]

        #             reshape_length = input_ids.shape[1] + add_tokens
        #         with nvtx.range("_sample part3.2.2:"):
        #             # all_input_ids_for_infer_merge = torch.zeros(input_ids.shape[0], input_ids.shape[1] + add_tokens, device=input_ids.device, dtype=input_ids.dtype)
        #             # 复用缓存，截取需要的长度（这一步几乎不耗时）
        #             all_input_ids_for_infer_merge = self.input_merge_cache[:, :reshape_length].to(input_ids.device)
        #         with nvtx.range("_sample part3.2.3:"):
        #             audio_in_token = 128015

        # with nvtx.range("_sample part4:"):
        #     for b_idx in range(input_ids.shape[0]):
        #         insert_pos = 0
        #         insert_pos_before_merge = 0
        #         audio_in_token_pos = (input_ids[b_idx] == audio_in_token).nonzero()[0]
        #         # if str(input_ids.device) == 'cuda:7':
        #         #     print(audio_in_token_pos, 'audio_in_token_pos\n')
        #         #     print(input_ids.shape, input_ids.shape, 'shape\n')
        #         for audio_idx, t_idx in enumerate(audio_in_token_pos):
        #             if audio_idx == 0:
        #                 all_input_ids_for_infer_merge[b_idx, :t_idx] = input_ids[b_idx, :t_idx]
        #                 insert_pos = t_idx
        #                 insert_pos_before_merge = t_idx
        #             else:
        #                 all_input_ids_for_infer_merge[b_idx, insert_pos:insert_pos + (t_idx - audio_in_token_pos[audio_idx - 1] - 1)] = input_ids[b_idx, audio_in_token_pos[audio_idx - 1] + 1:t_idx]
        #                 insert_pos = insert_pos + (t_idx - audio_in_token_pos[audio_idx - 1] - 1)
        #                 insert_pos_before_merge = insert_pos_before_merge + (t_idx - audio_in_token_pos[audio_idx - 1] - 1)
        #             # if str(all_input_ids_for_infer.device) == 'cuda:7':
        #             #     print(insert_pos_before_merge, t_idx, insert_pos, "insert_pos_before_merge, t_idx, insert_pos first\n")
        #             num_add_tokens = audio_features_length[audio_idx]
        #             all_input_ids_for_infer_merge[b_idx, insert_pos:insert_pos + num_add_tokens] = audio_in_token
        #             insert_pos = insert_pos + num_add_tokens
        #             insert_pos_before_merge = insert_pos_before_merge + 1
        #             # if str(all_input_ids_for_infer.device) == 'cuda:7':
        #             #     print(insert_pos_before_merge, t_idx, insert_pos, "insert_pos_before_merge, t_idx, insert_pos second\n")

        #             #     if str(all_input_ids_for_infer.device) == 'cuda:7':
        #             #         print(all_input_ids_for_infer_merge[b_idx, insert_pos:].shape, 'all_input_ids_for_infer_merge[b_idx, insert_pos:].shape\n')
        #             #         print(all_input_ids_for_infer[b_idx, insert_pos_before_merge:].shape, 'all_input_ids_for_infer[b_idx, insert_pos_before_merge:].shape\n')
        #             #         print(insert_pos, 'insert_pos\n')
        #             #         print(insert_pos_before_merge, 'insert_pos_before_merge\n')
        #         all_input_ids_for_infer_merge[b_idx, insert_pos:] = input_ids[b_idx, insert_pos_before_merge:]

        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
        ):
            # Check which multimodal stage we are in
            # FIXME: Assume single input generation
            if input_ids[0][-1] == audio_out_bos_token_id or input_ids[0][-1] == audio_out_last_bos_token_id:
                generation_mode = GenerationMode.AUDIO_INIT
            elif input_ids[0][-1] == self.audio_out_token_idx:
                generation_mode = GenerationMode.AUDIO_IN_PROGRESS
            else:
                generation_mode = GenerationMode.TEXT

            is_audio_generation_mode = generation_mode == GenerationMode.AUDIO_IN_PROGRESS

            if init_model_input or not generation_config.use_cache:
                model_inputs = {"input_ids": input_ids, **model_kwargs}
            else:
                model_inputs = {"input_ids": input_ids[:, -1:], **model_kwargs}

                if is_audio_generation_mode and generation_config.use_cache:
                    model_inputs["audio_out_ids"] = model_kwargs["audio_out_ids"][:, -1:]
                    model_inputs["audio_out_ids_start"] = torch.tensor([0], dtype=torch.long, device=input_ids.device)
                elif not is_audio_generation_mode:
                    del model_inputs["audio_out_ids"]
                    del model_inputs["audio_out_ids_start"]

                if generation_config.use_cache:
                    if "audio_features" in model_inputs and model_inputs["audio_features"] is not None:
                        model_inputs["audio_features"] = model_inputs["audio_features"][:0, ...]
                        model_inputs["audio_feature_attention_mask"] = model_inputs["audio_feature_attention_mask"][
                            :0, ...
                        ]

                    if "audio_in_ids" in model_inputs and model_inputs["audio_in_ids"] is not None:
                        model_inputs["audio_in_ids"] = None
                        model_inputs["audio_in_ids_start"] = None

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            if past_key_values_buckets is not None:
                past_key_values, self.current_past_key_values_bucket = self._prepare_kv_cache(
                    cur_len, self.current_past_key_values_bucket, past_key_values_buckets
                )
                if past_key_values is not None:
                    model_inputs.update({"past_key_values": past_key_values})
                model_inputs["past_key_values_buckets"] = past_key_values_buckets

            model_inputs["all_input_ids_for_infer"] = self.all_input_ids_for_infer

            # forward pass to get next token
            outputs = self(**model_inputs, return_dict=True)

            # Update the actual sequence length after the first forward pass
            if init_model_input and past_key_values_buckets is not None:
                cur_len = past_key_values_buckets[self.current_past_key_values_bucket].get_seq_length().item()

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
                extend_attention_mask=True,
            )

            # After the first forward pass, we can set init_model_input to False.
            init_model_input = False

            if synced_gpus and this_peer_finished:
                continue

            if is_audio_generation_mode:
                # In audio generation mode, we sample the audio tokens from audio logits.
                # It might also generate the audio eos token to end the audio generation.
                (
                    next_tokens,
                    next_audio_tokens,
                    next_audio_token_logits,
                    next_audio_token_scores,
                    num_delay,
                    num_remaining_delays,
                ) = self._sample_audio_tokens(
                    hidden_states=outputs.audio_hidden_states,
                    audio_logits=outputs.audio_logits,
                    audio_out_ids=model_kwargs["audio_out_ids"],
                    do_sample=do_sample,
                    logits_processor=logits_processor,
                    device=input_ids.device,
                    torch_generator=torch_generator,
                    generation_config=generation_config,
                    num_delay=num_delay,
                    num_remaining_delays=num_remaining_delays,
                )

                # update generated ids, model inputs, and length for next step
                model_kwargs["audio_out_ids"] = torch.cat(
                    [model_kwargs["audio_out_ids"], next_audio_tokens[:, None]], dim=-1
                )
                audio_sequences[-1] = torch.cat([audio_sequences[-1], next_audio_tokens[:, None]], dim=-1)

                if streamer is not None:
                    streamer.put(next_audio_tokens.cuda())
            else:
                # In text generation mode, we sample the text tokens from text logits.
                # It might also generate the audio placeholder token to start the audio generation.
                next_tokens, next_audio_tokens, next_token_logits, next_token_scores = self._sample_text_tokens(
                    input_ids=input_ids,
                    logits=outputs.logits,
                    do_sample=do_sample,
                    logits_processor=logits_processor,
                    device=input_ids.device,
                    generation_mode=generation_mode,
                    torch_generator=torch_generator,
                )

                if streamer is not None:
                    streamer.put(next_tokens.cuda())

                if next_audio_tokens is not None:
                    # If the token is audio bos token, we will generate the audio placeholder token
                    # and the corrensponding audio stream bos token to start the audio generation.
                    audio_sequences.append(next_audio_tokens[:, None])
                    if streamer is not None:
                        streamer.put(next_audio_tokens.cuda())
                    if model_kwargs["audio_out_ids"] is None or model_kwargs["audio_out_ids"].shape[0] == 0:
                        # Initialize audio_out_ids
                        model_kwargs["audio_out_ids"] = next_audio_tokens[:, None]
                        model_kwargs["audio_out_ids_start"] = torch.tensor(
                            [0], dtype=torch.long, device=input_ids.device
                        )
                    else:
                        model_kwargs["audio_out_ids_start"] = torch.concat(
                            [
                                model_kwargs["audio_out_ids_start"],
                                torch.tensor(
                                    [model_kwargs["audio_out_ids"].shape[1]], dtype=torch.long, device=input_ids.device
                                ),
                            ],
                            dim=0,
                        )
                        model_kwargs["audio_out_ids"] = torch.concat(
                            [model_kwargs["audio_out_ids"], next_audio_tokens[:, None]], dim=1
                        )

            if return_dict_in_generate:
                if output_scores:
                    if is_audio_generation_mode:
                        scores += (next_audio_token_scores,)
                    else:
                        scores += (next_token_scores,)
                if output_logits:
                    if is_audio_generation_mode:
                        raw_logits += (next_audio_token_logits,)
                    else:
                        raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (outputs.attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (outputs.hidden_states,)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            if "tokenizer_length" in generation_config.generation_kwargs:
                tokenizer_length = generation_config.generation_kwargs["tokenizer_length"]
                if torch.max(next_tokens) >= tokenizer_length:
                    raise ValueError(
                        f"Next generated token has max value {torch.max(next_tokens)} which is greater than the tokenizer's vocabulary size {tokenizer_length}, this is undesired behavior."
                    )

            # update generated ids, model inputs, and length for next step
            if not is_audio_generation_mode or next_tokens[0] != self.audio_out_token_idx:
                # We only add one <|AUDIO_OUT|> token to the input_ids for simplicity.
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            self.all_input_ids_for_infer = torch.cat([self.all_input_ids_for_infer, next_tokens[:, None]], dim=-1)
            input_ids_full = torch.cat([input_ids_full, next_tokens[:, None]], dim=-1)
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids_full, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        return_dict_in_generate = False
        if return_dict_in_generate:
            return HiggsAudioGenerationOutput(
                sequences=input_ids,
                audio_sequences=audio_sequences,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return input_ids, audio_sequences, raw_logits

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_feature_attention_mask: Optional[torch.BoolTensor] = None,
        audio_in_ids: Optional[torch.LongTensor] = None,
        audio_in_ids_start: Optional[torch.LongTensor] = None,
        audio_out_ids: Optional[torch.LongTensor] = None,
        audio_out_ids_start: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        audio_out_bos_token_id: int = None,
        audio_eos_token_id: int = None,
        past_key_values_buckets: Optional[OrderedDict[int, Cache]] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        The generate function in huggingface generally follows these steps:

        for sample_step in 1, 2, 3, 4, 5, ...
            ...

        """
        with nvtx.range("before super.generate()"):
            # Right now, it's a very simplified version of generate, we should revisit this after our model architecture stabilizes.
            assert input_ids.shape[0] == 1, (
                "Currently HiggsAudioModel.generate() only supports batch_size=1. See the implementation of "
            )

            stop_strings = kwargs.pop("stop_strings", [])

            generation_config, kwargs = self._prepare_generation_config(kwargs.pop("generation_config", None), **kwargs)
            if audio_out_bos_token_id is not None:
                generation_config.generation_kwargs["audio_out_bos_token_id"] = audio_out_bos_token_id
            else:
                try:
                    generation_config.generation_kwargs["audio_out_bos_token_id"] = self.audio_out_bos_token_id
                except:
                    generation_config.generation_kwargs["audio_out_bos_token_id"] = None

            if audio_eos_token_id is not None:
                generation_config.generation_kwargs["audio_eos_token_id"] = audio_eos_token_id
            else:
                try:
                    generation_config.generation_kwargs["audio_eos_token_id"] = self.audio_eos_token_id
                except:
                    generation_config.generation_kwargs["audio_eos_token_id"] = None

            has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
            has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None

            generation_config.generation_kwargs["ras_win_len"] = kwargs.pop("ras_win_len", None)
            generation_config.generation_kwargs["ras_win_max_num_repeat"] = kwargs.pop("ras_win_max_num_repeat", 2)
            # Set generation seed if determinstic generation is required
            if seed is not None:
                generation_config.generation_kwargs["seed"] = seed

            # Store tokenizer in generation config if it is in kwargs without popping it
            if "tokenizer" in kwargs:
                generation_config.generation_kwargs["tokenizer_length"] = len(kwargs["tokenizer"])

            # input_ids: [bsz, seq_len]
            # The merging of audio features happens inside the forward path. The input_ids does not need to change.
            # TODO: prepare the final input embeddings to improve generation performance
            input_ids_length = input_ids.shape[-1]
            generation_config = self._prepare_generated_length(
                generation_config=generation_config,
                has_default_max_length=has_default_max_length,
                has_default_min_length=has_default_min_length,
                model_input_name=None,
                inputs_tensor=None,
                input_ids_length=input_ids_length,
            )
            assert generation_config.num_beams == 1, "Currently, we only support beam search with num_beams=1"
            return_dict_in_generate = generation_config.return_dict_in_generate
            output_scores = generation_config.output_scores

            # When attn_implement is spda or flash-attention, it will create causal mask automatically.
            attention_mask = kwargs.pop("attention_mask", None)

        return super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_features=audio_features,
            audio_feature_attention_mask=audio_feature_attention_mask,
            audio_in_ids=audio_in_ids,
            audio_in_ids_start=audio_in_ids_start,
            audio_out_ids=audio_out_ids,
            audio_out_ids_start=audio_out_ids_start,
            past_key_values=past_key_values,
            generation_config=generation_config,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            past_key_values_buckets=past_key_values_buckets,
            **kwargs,
        )

    def parameter_count_per_component(self):
        """Count the number of parameters per component in the model.

        HiggsAudio has the following main components:
            audio_tower: For mapping audio features to hidden states),
            llm_embed: The size of embedding layer of the LLM
            llm_non_embed: The size of non-embedding layer of the LLM
            audio_adapter: The overall size of additional layers for audio generation

        """
        trainable_stats = {
            "audio_tower": 0,
            "llm_embed": 0,
            "llm_non_embed": 0,
            "audio_embed": 0,
            "audio_adapter": 0,
            "overall": 0,
        }
        total_stats = {
            "audio_tower": 0,
            "llm_embed": 0,
            "llm_non_embed": 0,
            "audio_embed": 0,
            "audio_adapter": 0,
            "overall": 0,
        }

        total_stats["overall"] = count_parameters(self, trainable_only=False)
        trainable_stats["overall"] = count_parameters(self, trainable_only=True)

        for mod in [self.audio_tower]:
            if mod is not None:
                total_stats["audio_tower"] += count_parameters(mod, trainable_only=False)
                trainable_stats["audio_tower"] += count_parameters(mod, trainable_only=True)

        total_stats["llm_embed"] = count_parameters(self.embed_tokens, trainable_only=False)
        trainable_stats["llm_embed"] = count_parameters(self.embed_tokens, trainable_only=True)

        total_stats["audio_embed"] = count_parameters(self.audio_codebook_embeddings, trainable_only=False)
        trainable_stats["audio_embed"] = count_parameters(self.audio_codebook_embeddings, trainable_only=True)

        # Calculate number of parameters for LLM
        for layer in self.layers:
            if isinstance(layer, HiggsAudioDualFFNDecoderLayer):
                total_param_count = count_parameters(layer, trainable_only=False)
                total_trainable_param_count = count_parameters(layer, trainable_only=True)
                total_stats["llm_non_embed"] += total_param_count
                trainable_stats["llm_non_embed"] += total_trainable_param_count
                if not layer.fast_forward:
                    audio_mlp_param_count = count_parameters(layer.audio_mlp, trainable_only=False)
                    audio_mlp_trainable_param_count = count_parameters(layer.audio_mlp, trainable_only=True)

                    audio_norm_param_count = count_parameters(
                        layer.audio_post_attention_layernorm, trainable_only=False
                    ) + count_parameters(layer.audio_input_layernorm, trainable_only=False)
                    audio_norm_trainable_param_count = count_parameters(
                        layer.audio_post_attention_layernorm, trainable_only=True
                    ) + count_parameters(layer.audio_input_layernorm, trainable_only=True)
                    total_stats["llm_non_embed"] -= audio_mlp_param_count + audio_norm_param_count
                    trainable_stats["llm_non_embed"] -= (
                        audio_mlp_trainable_param_count + audio_norm_trainable_param_count
                    )
                    total_stats["audio_adapter"] += audio_mlp_param_count + audio_norm_param_count
                    trainable_stats["audio_adapter"] += (
                        audio_mlp_trainable_param_count + audio_norm_trainable_param_count
                    )

                    if layer.use_audio_attention:
                        audio_attn_param_count = count_parameters(
                            layer.audio_attn, trainable_only=False
                        ) + count_parameters(layer.audio_post_audio_attn_layer_norm, trainable_only=False)
                        audio_attn_trainable_param_count = count_parameters(
                            layer.audio_attn, trainable_only=True
                        ) + count_parameters(layer.audio_post_audio_attn_layer_norm, trainable_only=True)
                        total_stats["llm_non_embed"] -= audio_attn_param_count
                        trainable_stats["llm_non_embed"] -= audio_attn_trainable_param_count
                        total_stats["audio_adapter"] += audio_attn_param_count
                        trainable_stats["audio_adapter"] += audio_attn_trainable_param_count
            else:
                total_stats["llm_non_embed"] += count_parameters(layer, trainable_only=False)
                trainable_stats["llm_non_embed"] += count_parameters(layer, trainable_only=True)
        total_stats["llm_non_embed"] += count_parameters(self.norm, trainable_only=False)
        trainable_stats["llm_non_embed"] += count_parameters(self.norm, trainable_only=True)

        total_stats["audio_adapter"] += count_parameters(self.audio_decoder_proj.audio_lm_head, trainable_only=False)
        trainable_stats["audio_adapter"] += count_parameters(
            self.audio_decoder_proj.audio_lm_head, trainable_only=True
        )
        total_stats["llm_embed"] += count_parameters(self.audio_decoder_proj.text_lm_head, trainable_only=False)
        trainable_stats["llm_embed"] += count_parameters(self.audio_decoder_proj.text_lm_head, trainable_only=True)

        other_audio_modules = [self.audio_encoder_proj]
        if self.use_audio_out_embed_projector:
            other_audio_modules.append(self.audio_out_embed_projector)

        for mod in other_audio_modules:
            if mod is not None:
                total_stats["audio_adapter"] += count_parameters(mod, trainable_only=False)
                trainable_stats["audio_adapter"] += count_parameters(mod, trainable_only=True)
        return {"trainable": trainable_stats, "total": total_stats}

    def set_unskip_audio_tower(self):
        self.config.skip_audio_tower = False
        self.config.encode_whisper_embed = True

    def set_skip_audio_tower(self):
        self.config.skip_audio_tower = True
        self.config.encode_whisper_embed = False

    def set_encode_audio_in_tokens(self):
        self.config.encode_audio_in_tokens = True

    def freeze_audio_tower(self):
        if self.audio_tower is not None:
            for param in self.audio_tower.parameters():
                param.requires_grad = False

    def freeze_audio_encoder_proj(self):
        if self.audio_encoder_proj is not None:
            for param in self.audio_encoder_proj.parameters():
                param.requires_grad = False

    def freeze_llm(self, freeze_embed=True, freeze_embed_until_idx: Optional[int] = None, freeze_text_added_tokens=True):
        for layer in self.layers:
            if isinstance(layer, HiggsAudioDualFFNDecoderLayer):
                for param in layer.self_attn.parameters():
                    param.requires_grad = False
                for param in layer.mlp.parameters():
                    param.requires_grad = False

                for param in layer.post_attention_layernorm.parameters():
                    param.requires_grad = False

                for param in layer.input_layernorm.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = False

        for param in self.norm.parameters():
            param.requires_grad = False

        if freeze_embed:
            if freeze_embed_until_idx is None:
                for param in self.embed_tokens.parameters():
                    param.requires_grad = False
                for param in self.audio_decoder_proj.text_lm_head.parameters():
                    param.requires_grad = False
            else:
                assert isinstance(self.embed_tokens, nn.Embedding)
                self.embed_tokens = PartiallyFrozenEmbedding(
                    original_embedding=self.embed_tokens, freeze_until_idx=freeze_embed_until_idx
                )

        # if not freeze_text_added_tokens:
        #     self.embed_tokens.weight[[i for i in range(128000, 128021)]].requires_grad = True

        #     self.audio_decoder_proj.text_lm_head.weight[:, 128000: 128021].requires_grad = True
        # else:
        #     self.embed_tokens.weight[[i for i in range(128012, 128014)]].requires_grad = True
        #     self.embed_tokens.weight[[i for i in range(128016, 128017)]].requires_grad = True

        #     self.audio_decoder_proj.text_lm_head.weight[:, 128012: 128014].requires_grad = True
        #     self.audio_decoder_proj.text_lm_head.weight[:, 128016: 128017].requires_grad = True


    def freeze_text_head(self, freeze_text_head_until_idx: Optional[int] = None):
        """Freeze the final text head"""
        if freeze_text_head_until_idx is None:
            for param in self.audio_decoder_proj.text_lm_head.parameters():
                param.requires_grad = False

        else:
            assert isinstance(self.audio_decoder_proj.text_lm_head, nn.Linear)
            self.audio_decoder_proj.text_lm_head = PartiallyFrozenLinear(
                original_linear=self.audio_decoder_proj.text_lm_head, freeze_until_idx=freeze_text_head_until_idx
            )

    def freeze_all(self, freeze_audio_lm_head=False):
        for param in self.parameters():
            param.requires_grad = False

        if not freeze_audio_lm_head:
            for param in self.audio_decoder_proj.audio_lm_head.parameters():
                param.requires_grad = True

    @classmethod
    def merge_weights_from_checkpoint(cls, checkpoint_dir: str, merged_output_dir: str, *model_args, **kwargs):
        # For users' convenience, we merge back embedding and text_lm_head if they are splitted
        splitted_model = super().from_pretrained(
            checkpoint_dir,
            *model_args,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            **{**kwargs, "state_dict": None},  # Prevent auto-loading state_dict
        )

        # Load all safetensor shards
        state_dict = {}
        shard_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))

        for shard_path in shard_paths:
            shard_dict = load_file(shard_path)  # Load each shard
            state_dict.update(shard_dict)  # Merge into a single dict

        # Merge weights
        if (
            "audio_decoder_proj.text_lm_head.linear_frozen.weight" in state_dict
            and "audio_decoder_proj.text_lm_head.linear_trainable.weight" in state_dict
        ):
            state_dict["audio_decoder_proj.text_lm_head.weight"] = torch.cat(
                [
                    state_dict["audio_decoder_proj.text_lm_head.linear_frozen.weight"],
                    state_dict["audio_decoder_proj.text_lm_head.linear_trainable.weight"],
                ],
                dim=0,
            )

            del state_dict["audio_decoder_proj.text_lm_head.linear_frozen.weight"]
            del state_dict["audio_decoder_proj.text_lm_head.linear_trainable.weight"]

        if (
            "embed_tokens.embedding_frozen.weight" in state_dict
            and "embed_tokens.embedding_trainable.weight" in state_dict
        ):
            state_dict["embed_tokens.weight"] = torch.cat(
                [
                    state_dict["embed_tokens.embedding_frozen.weight"],
                    state_dict["embed_tokens.embedding_trainable.weight"],
                ],
                dim=0,
            )

            del state_dict["embed_tokens.embedding_frozen.weight"]
            del state_dict["embed_tokens.embedding_trainable.weight"]

        # Load the final state_dict
        splitted_model.load_state_dict(state_dict, strict=True)

        if merged_output_dir:
            splitted_model.save_pretrained(merged_output_dir, is_main_process=True, state_dict=state_dict)

    @torch.inference_mode()
    def capture_model(self, past_key_values: list[Union[Cache, List[torch.FloatTensor]]]) -> None:
        """Capture CUDA graphs for the model's forward pass with different KV cache lengths.

        Args:
            past_key_values: List of KV caches to capture graphs for
        """
        for past_key_value in past_key_values:
            kv_cache_length = past_key_value.get_max_cache_shape()
            # We capture two graphs, one for decoding audio tokens and one for decoding text tokens
            for is_decoding_audio_token in [True, False]:
                runner = CUDAGraphRunner(self._forward_core)

                # Create dummy inputs for graph capture
                batch_size = 1
                hidden_dim = self.config.hidden_size

                hidden_states = torch.zeros(
                    (batch_size, 1, hidden_dim), dtype=self.config.torch_dtype, device=self.device
                )
                causal_mask = torch.ones(
                    (batch_size, 1, 1, kv_cache_length), dtype=self.config.torch_dtype, device=self.device
                )
                position_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
                audio_discrete_codes_mask = torch.tensor(
                    [[is_decoding_audio_token]], dtype=torch.bool, device=self.device
                )
                cache_position = torch.tensor([kv_cache_length - 1], dtype=torch.long, device=self.device)
                audio_attention_mask = torch.ones_like(causal_mask)
                fast_forward_attention_mask = torch.ones_like(causal_mask)

                runner.capture(
                    hidden_states=hidden_states,
                    text_attention_mask=causal_mask,
                    position_ids=position_ids,
                    audio_discrete_codes_mask=audio_discrete_codes_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_value,
                    use_cache=True,
                    audio_attention_mask=audio_attention_mask,
                    fast_forward_attention_mask=fast_forward_attention_mask,
                    output_attentions=False,
                    output_hidden_states=False,
                    is_decoding_audio_token=is_decoding_audio_token,
                    is_using_cuda_graph=True,
                    stream=torch.cuda.Stream(device=self.device),
                )

                self.decode_graph_runners[kv_cache_length][is_decoding_audio_token] = runner
