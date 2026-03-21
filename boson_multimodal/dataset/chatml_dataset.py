import dacite
import pandas as pd
import torch
import json

import numpy as np
import multiprocessing as mp

from dataclasses import dataclass, fields
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Optional

from ..data_types import ChatMLSample, TextContent, AudioContent
from ..constants import AUDIO_IN_TOKEN, AUDIO_OUT_TOKEN

from loguru import logger

# Whisper processor, 30 sec -> 3000 features
# Then we divide 4 in the audio towker, we decrease 3000 features to 750, which gives 25 Hz
WHISPER_EMBED_NUM_HIDDEN_STATE_PER_SEC = 25


@dataclass
class ChatMLDatasetSample:
    input_ids: torch.LongTensor  # Shape (seq_len,): The input text tokens.
    label_ids: torch.LongTensor  # Shape (seq_len,): The label ids.
    audio_ids_concat: torch.LongTensor  # Shape (num_codebooks, audio_seq_len): The audio tokens that are concatenated.
    # Here `audio_seq_len` is the length of the concatenated audio tokens.`
    audio_ids_start: (
        torch.LongTensor
    )  # Shape (num_audios,): The start index of each audio token in the concatenated audio tokens.
    audio_waveforms_concat: (
        torch.Tensor
    )  # Shape (total_wv_length,): The concatenated audio waveforms for audio-in features.
    audio_waveforms_start: (
        torch.LongTensor
    )  # Shape (num_audios,): The start index of each audio waveform in the concatenated audio waveforms.
    audio_sample_rate: torch.Tensor  # Shape (num_audios,): The sampling rate of the audio waveforms.
    audio_speaker_indices: (
        torch.LongTensor
    )  # Shape (num_audios,) -1 means unknown speaker: The speaker indices for each audio.
    audio_label_ids_concat: Optional[torch.LongTensor] = (
        None  # Shape (num_codebooks, audio_seq_len): The audio tokens that are concatenated.
    )
    # Here `audio_seq_len` is the length of the concatenated audio tokens.`
    label_audio_ids: Optional[torch.LongTensor] = None  # 新增字段
    reward: Optional[float] = None

    def num_audios(self):
        return max(len(self.audio_waveforms_start), len(self.audio_ids_start))

    def get_audio_codes(self, idx):
        code_start = self.audio_ids_start[idx]
        if idx < len(self.audio_ids_start) - 1:
            code_end = self.audio_ids_start[idx + 1]
        else:
            code_end = self.audio_ids_concat.shape[-1]

        return self.audio_ids_concat[:, code_start:code_end]

    def get_audio_codes_labels(self, idx):
        if self.audio_label_ids_concat is None:
            return None
        code_start = self.audio_ids_start[idx]
        if idx < len(self.audio_ids_start) - 1:
            code_end = self.audio_ids_start[idx + 1]
        else:
            code_end = self.audio_ids_concat.shape[-1]

        return self.audio_label_ids_concat[:, code_start:code_end]

    def get_wv(self, idx):
        wv_start = self.audio_waveforms_start[idx]
        sr = self.audio_sample_rate[idx]
        if idx < len(self.audio_waveforms_start) - 1:
            wv_end = self.audio_waveforms_start[idx + 1]
        else:
            wv_end = self.audio_waveforms_concat.shape[-1]
        return self.audio_waveforms_concat[wv_start:wv_end], sr

    def cal_num_tokens(
        self,
        encode_whisper_embed: bool = True,
        encode_audio_in_tokens: bool = False,
        encode_audio_out_tokens: bool = True,
        audio_in_token_id: int = 128015,
        audio_out_token_id: int = 128016,
    ) -> int:
        # we firstly exclude <|AUDIO|> and <|AUDIO_OUT|> because we do late merging and replace those position with actual audio features and audio token ids
        # It's assumed that we always have audio_ids when audio_waveforms are there (but not vice-versa)
        num_tokens = len(self.input_ids) - len(self.audio_ids_start)

        if encode_whisper_embed and len(self.audio_waveforms_concat) > 0:
            audio_lengths = torch.diff(self.audio_waveforms_start)
            if len(audio_lengths):
                # Sum before calling .item()
                num_tokens += (
                    (
                        np.ceil(WHISPER_EMBED_NUM_HIDDEN_STATE_PER_SEC * audio_lengths / self.audio_sample_rate[:-1])
                    ).sum()
                ).item()
            # add the last audio's token estimation
            num_tokens += (
                np.ceil(
                    WHISPER_EMBED_NUM_HIDDEN_STATE_PER_SEC
                    * (self.audio_waveforms_concat.shape[0] - self.audio_waveforms_start[-1])
                    / self.audio_sample_rate[-1]
                )
            ).item()

        if self.audio_ids_concat.size(1) > 0:
            audio_io_ids = self.input_ids[
                (self.input_ids == audio_in_token_id) | (self.input_ids == audio_out_token_id)
            ]
            audio_io_id_lengths = torch.concat(
                [
                    torch.diff(self.audio_ids_start),
                    torch.tensor([self.audio_ids_concat.shape[-1] - self.audio_ids_start[-1]]),
                ]
            )
            if encode_audio_in_tokens:
                num_tokens += torch.sum(audio_io_id_lengths[audio_io_ids == audio_in_token_id]).item()

            if encode_audio_out_tokens:
                num_tokens += torch.sum(audio_io_id_lengths[audio_io_ids == audio_out_token_id]).item()

        return int(num_tokens)

    @classmethod
    def merge(
        cls,
        samples: List["ChatMLDatasetSample"],
        eos_token_id: int,
        ignore_index: int,
        padding_size: Optional[int] = None,
    ) -> "ChatMLDatasetSample":
        """Merges a list of ChatMLDatasetSample instances, inserting eos_token_id and ignore_index between them, and adjusting offsets for audio_ids_start and audio_waveforms_start.

        Args:
            samples (List[ChatMLDatasetSample]): List of samples to merge.
            eos_token_id (int): Tokens to be inserted into input_ids between samples.
            ignore_index (int): Default label for padding.
            padding_size (Optional[int]): If provided, pad the sequence to with this length.

        Returns:
            ChatMLDatasetSample: Merged and potentially padded sample.
        """
        if not samples:
            logger.fatal("The samples list is empty and cannot be merged.")
            raise ValueError("The samples list is empty and cannot be merged.")

        # Initialize empty lists for concatenation
        input_ids_list = []
        label_ids_list = []
        audio_ids_concat_list = []
        audio_ids_start_list = []
        audio_waveforms_concat_list = []
        audio_waveforms_start_list = []
        audio_sample_rate_list = []
        audio_speaker_indices_list = []
        label_audio_ids_list = []  # 新增

        # Track offsets
        audio_ids_offset = 0
        audio_waveforms_offset = 0

        for sample in samples:
            # Add input_ids and label_ids with padding
            if input_ids_list:
                input_ids_list.append(torch.tensor([eos_token_id], dtype=torch.long))
                label_ids_list.append(torch.tensor([ignore_index], dtype=torch.long))
            input_ids_list.append(sample.input_ids)
            label_ids_list.append(sample.label_ids)

            # Add audio_ids_concat and handle empty audio ids
            if sample.audio_ids_concat.size(1) > 0:
                audio_ids_concat_list.append(sample.audio_ids_concat)

                # Offset and add audio_ids_start
                audio_ids_start_list.append(sample.audio_ids_start + audio_ids_offset)
                audio_ids_offset += sample.audio_ids_concat.size(
                    1
                )  # (num_codebooks, seq_len): Update offset by audio_seq_len

            # Add label_audio_ids
            if sample.label_audio_ids is not None:
                label_audio_ids_list.append(sample.label_audio_ids)

            # Add audio_waveforms_concat
            if sample.audio_waveforms_concat.size(0) > 0:
                # Check dimensions of the audio waveform to ensure consistency
                if (
                    audio_waveforms_concat_list
                    and sample.audio_waveforms_concat.dim() != audio_waveforms_concat_list[0].dim()
                ):
                    logger.warning(
                        f"Skipping audio waveform with inconsistent dimensions: expected {audio_waveforms_concat_list[0].dim()}D, got {sample.audio_waveforms_concat.dim()}D"
                    )
                    continue

                audio_waveforms_concat_list.append(sample.audio_waveforms_concat)
                audio_waveforms_start_list.append(sample.audio_waveforms_start + audio_waveforms_offset)
                audio_waveforms_offset += sample.audio_waveforms_concat.size(0)

                # Add audio_sample_rate and audio_speaker_indices
                audio_sample_rate_list.append(sample.audio_sample_rate)

            audio_speaker_indices_list.append(sample.audio_speaker_indices)

        # Concatenate all tensors
        input_ids = torch.cat(input_ids_list, dim=0)
        label_ids = torch.cat(label_ids_list, dim=0)

        # Apply padding if padding_size is specified
        if padding_size is not None and padding_size > 0:
            input_ids = torch.cat([input_ids, torch.full((padding_size,), eos_token_id, dtype=torch.long)], dim=0)
            label_ids = torch.cat([label_ids, torch.full((padding_size,), ignore_index, dtype=torch.long)], dim=0)

        # Safely concatenate audio tensors with proper error handling
        try:
            audio_ids_concat = torch.cat(audio_ids_concat_list, dim=1) if audio_ids_concat_list else torch.tensor([[]])
            audio_ids_start = torch.cat(audio_ids_start_list, dim=0) if audio_ids_start_list else torch.tensor([])
            label_audio_ids = torch.cat(label_audio_ids_list, dim=-1) if label_audio_ids_list else None

            # Check for dimensional consistency in audio waveforms
            if audio_waveforms_concat_list:
                dims = [t.dim() for t in audio_waveforms_concat_list]
                if not all(d == dims[0] for d in dims):
                    # If dimensions don't match, log warning and filter out the problematic tensors
                    logger.warning(
                        f"Inconsistent dimensions in audio waveforms: {dims}. Filtering to keep only consistent ones."
                    )
                    expected_dim = max(set(dims), key=dims.count)  # Most common dimension
                    audio_waveforms_concat_list = [t for t in audio_waveforms_concat_list if t.dim() == expected_dim]

                    # Recalculate audio_waveforms_start with the filtered list
                    if audio_waveforms_concat_list:
                        audio_waveforms_offset = 0
                        audio_waveforms_start_list = []
                        for waveform in audio_waveforms_concat_list:
                            audio_waveforms_start_list.append(torch.tensor([audio_waveforms_offset]))
                            audio_waveforms_offset += waveform.size(0)

            audio_waveforms_concat = (
                torch.cat(audio_waveforms_concat_list, dim=0) if audio_waveforms_concat_list else torch.tensor([])
            )
            audio_waveforms_start = (
                torch.cat(audio_waveforms_start_list, dim=0) if audio_waveforms_start_list else torch.tensor([])
            )
            audio_sample_rate = (
                torch.cat(audio_sample_rate_list, dim=0) if audio_sample_rate_list else torch.tensor([])
            )
            audio_speaker_indices = (
                torch.cat(audio_speaker_indices_list, dim=0) if audio_speaker_indices_list else torch.tensor([])
            )

        except RuntimeError as e:
            logger.error(f"Error during tensor concatenation: {str(e)}")
            logger.warning("Falling back to empty audio tensors")
            # Fall back to empty tensors
            audio_ids_concat = torch.tensor([[]])
            audio_ids_start = torch.tensor([])
            audio_waveforms_concat = torch.tensor([])
            audio_waveforms_start = torch.tensor([])
            audio_sample_rate = torch.tensor([])
            audio_speaker_indices = torch.tensor([])
            label_audio_ids = None

        # 在合并时也复制
        label_audio_ids = audio_ids_concat if audio_ids_concat.numel() > 0 else None

        merged_sample = cls(
            input_ids=input_ids,
            label_ids=label_ids,
            audio_ids_concat=audio_ids_concat,
            audio_ids_start=audio_ids_start,
            audio_waveforms_concat=audio_waveforms_concat,
            audio_waveforms_start=audio_waveforms_start,
            audio_sample_rate=audio_sample_rate,
            audio_speaker_indices=audio_speaker_indices,
            audio_label_ids_concat=audio_label_ids_concat,  # 原来的字段
            label_audio_ids=audio_ids_concat,  # 直接复制
        )


        return merged_sample


@dataclass
class RankedChatMLDatasetSampleTuple:
    samples: List[ChatMLDatasetSample]
    scores: List[float]

    def max_score_sample(self) -> ChatMLDatasetSample:
        idx = self.scores.index(max(self.scores))
        self.samples[idx].reward = self.scores[idx]
        return self.samples[idx]

    def min_score_sample(self) -> ChatMLDatasetSample:
        idx = self.scores.index(min(self.scores))
        self.samples[idx].reward = self.scores[idx]
        return self.samples[idx]


@dataclass
class ChatMLDatasetStorageSample:
    input_tokens: torch.LongTensor
    label_tokens: torch.LongTensor
    audio_bytes_cache_dir_index: int
    audio_codes_cache_dir_index: int
    audio_bytes_indices: torch.LongTensor
    audio_codes_indices: torch.LongTensor
    speaker_indices: torch.LongTensor
    file_index: int
    original_sample_index: int


def encode_audio_to_tokens(audio_content: AudioContent):
    """
    将音频内容编码为token IDs
    这是一个示例函数，需要根据你的实际音频编码器实现
    """
    # 如果音频内容已经包含编码后的tokens
    if hasattr(audio_content, 'codes') and audio_content.codes is not None:
        if isinstance(audio_content.codes, torch.Tensor):
            return audio_content.codes
        else:
            return torch.tensor(audio_content.codes, dtype=torch.long)
    
    # 如果没有预编码的tokens，返回空tensor
    logger.warning(f"Cannot encode audio content: {audio_content}")
    return torch.tensor([], dtype=torch.long)


# TODO(sxjscience): We need to revist the logic about parsing speaker ids.
# Currently, we assume that the speaker id is stored at the "misc" field in ChatMLSample.
def prepare_chatml_sample(sample: Union[ChatMLSample, Dict], tokenizer):
    """Preprocess the ChatML sample to get the tokens for the text part.

    Args:
        sample (ChatMLSample): The ChatML sample to preprocess.
        tokenizer: The tokenizer to use for encoding the text.

    """

    try:
        if not isinstance(sample, ChatMLSample):
            # Handle all fields that could be NaN
            if "speaker" in sample and pd.isna(sample["speaker"]):
                sample["speaker"] = None
            if "start_index" in sample and pd.isna(sample["start_index"]):
                sample["start_index"] = None
            if "content" in sample and pd.isna(sample["content"]):
                sample["content"] = ""

            # Convert any other potential NaN values in nested structures
            def convert_nan_to_none(obj):
                import numpy as np

                if isinstance(obj, (pd.Series, np.ndarray)):
                    return obj.tolist()
                elif pd.api.types.is_scalar(obj) and pd.isna(obj):
                    return None
                elif isinstance(obj, dict):
                    return {k: convert_nan_to_none(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):  # Fixed: Handle both list and tuple
                    return [convert_nan_to_none(item) for item in obj]
                return obj

            # Clean the sample data
            clean_sample = convert_nan_to_none(sample)

            val_keys = []
            for field in fields(ChatMLSample):
                if field.name in clean_sample:
                    val_keys.append(field.name)
            clean_sample = {k: clean_sample[k] for k in val_keys}

            try:
                sample = dacite.from_dict(
                    data_class=ChatMLSample, data=clean_sample, config=dacite.Config(strict=True, check_types=True)
                )
            except Exception as e:
                print(f"Failed to convert to ChatMLSample: {e}")
                print(f"Clean sample: {json.dumps(clean_sample, indent=2)}")
                return None, None, None, None, None

        input_tokens = []
        label_tokens = []
        audio_contents = []
        audio_label_contents = []  # 新增：收集音频标签
        speaker_id = None
        if sample.speaker is not None:
            speaker_id = sample.speaker
        elif sample.misc is not None:
            if "speaker" in sample.misc:
                speaker_id = sample.misc["speaker"]

        total_m = len(sample.messages)
        for turn_id, message in enumerate(sample.messages):
            role = message.role
            recipient = message.recipient
            content = message.content
            content_l = []

            if isinstance(content, str):
                content_l.append(TextContent(text=content))
            elif isinstance(content, TextContent):
                content_l.append(content)
            elif isinstance(content, AudioContent):
                content_l.append(content)
            elif isinstance(content, list):
                for ele in content:
                    if isinstance(ele, str):
                        content_l.append(TextContent(text=ele))
                    else:
                        content_l.append(ele)
            if turn_id == 0:
                prefix = f"<|begin_of_text|><|start_header_id|>{role}<|end_header_id|>\n\n"
            else:
                prefix = f"<|start_header_id|>{role}<|end_header_id|>\n\n"
            eot_postfix = "<|eot_id|>"
            eom_postfix = "<|eom_id|>"

            ta_response_postfix = "<|response_ta|>"
            t_response_postfix = "<|response_t|>"

            prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
            input_tokens.extend(prefix_tokens)
            label_tokens.extend([-100 for _ in prefix_tokens])

            if recipient:
                assert role == "assistant", "Recipient is only available for assistant role."
                recipient_tokens = tokenizer.encode(f"{recipient}<|recipient|>", add_special_tokens=False)
                input_tokens.extend(recipient_tokens)
                label_tokens.extend(recipient_tokens)

            for content in content_l:
                if content.type == "text":
                    text_tokens = tokenizer.encode(content.text, add_special_tokens=False)
                    input_tokens.extend(text_tokens)
                    if role == "assistant" and (sample.start_index is None or turn_id >= sample.start_index):
                        label_tokens.extend(text_tokens)
                    else:
                        label_tokens.extend([-100 for _ in text_tokens])

                elif content.type == "audio":
                    audio_contents.append(content)
                    
                    # label
                    if role == "assistant" and (sample.start_index is None or turn_id >= sample.start_index):
                        audio_label_contents.append(content)
                    # else:
                    #     audio_label_contents.append(None)
                    
                    if role == "user" or role == "system":
                        # Add the text tokens
                        text_tokens = tokenizer.encode(
                            f"<|audio_bos|><|AUDIO|><|audio_eos|>",
                            add_special_tokens=False,
                        )# [128011, 128015, 128012]
                        input_tokens.extend(text_tokens)
                        label_tokens.extend([-100 for _ in text_tokens])
                    elif role == "assistant":
                        # Add the text tokens for audio-out part.
                        text_tokens = tokenizer.encode(
                            f"<|audio_out_bos|><|AUDIO_OUT|><|audio_out_eos|>",
                            add_special_tokens=False,
                        )# [128013, 128016, 128012]
                        input_tokens.extend(text_tokens)
                        if sample.start_index is None or turn_id >= sample.start_index:
                            label_tokens.extend(text_tokens)
                        else:
                            label_tokens.extend([-100 for _ in text_tokens])
            next_id = turn_id + 1
            if role == "assistant" and next_id != total_m and sample.messages[next_id].role == "assistant":
                postfix_tokens = tokenizer.encode(eom_postfix, add_special_tokens=False)
                input_tokens.extend(postfix_tokens)
            else:
                # if role == "user":
                #     assert message.response_type == "ta" or message.response_type == "t", "Response type should be \"ta\" or \"t\"."
                #     if message.response_type == "ta":
                #         response_postfix_tokens = tokenizer.encode(ta_response_postfix, add_special_tokens=False)
                #     else:
                #         response_postfix_tokens = tokenizer.encode(t_response_postfix, add_special_tokens=False)
                #     input_tokens.extend(response_postfix_tokens)

                postfix_tokens = tokenizer.encode(eot_postfix, add_special_tokens=False)
                input_tokens.extend(postfix_tokens)
            if role == "assistant" and (sample.start_index is None or turn_id >= sample.start_index):
                label_tokens.extend(postfix_tokens)
            else:
                # if role == "user":
                #     label_tokens.extend([-100 for _ in response_postfix_tokens])
                label_tokens.extend([-100 for _ in postfix_tokens])

        return input_tokens, label_tokens, audio_contents, audio_label_contents, speaker_id

    except Exception as e:
        print(f"Error in prepare_chatml_sample: {str(e)}")
        print(f"Sample data: {json.dumps(sample, indent=2)}")
        return None, None, None, None, None

def prepare_chatml_sample_interval(
    sample: Union[ChatMLSample, Dict], 
    tokenizer,
    text_split_config=[5, 20],  # 新增：2元素配置列表[首段文本长, 后续每段文本长]（控制文本拆分）
):
    """Preprocess the ChatML sample to get the tokens for the text part.
    关键变更：仅拆分assistant文本为多段，每段后加audio_out标记；audio_contents保持1个完整音频路径。
    Args:
        text_split_config (List[int]): 2-element config list:
            [first_text_len, subsequent_text_len]
            - first_text_len: Token length of text in the first text-audio pair
            - subsequent_text_len: Token length of text in subsequent text-audio pairs
    """
    # 校验配置合法性
    assert len(text_split_config) == 2, "audio_text_len_config must be 2 elements"
    first_text_len, subsequent_text_len = text_split_config
    assert all(isinstance(x, int) and x > 0 for x in text_split_config), "All config values must be positive integers"

    try:
        if not isinstance(sample, ChatMLSample):
            # 原有NaN处理逻辑完全保留
            if "speaker" in sample and pd.isna(sample["speaker"]):
                sample["speaker"] = None
            if "start_index" in sample and pd.isna(sample["start_index"]):
                sample["start_index"] = None
            if "content" in sample and pd.isna(sample["content"]):
                sample["content"] = ""

            def convert_nan_to_none(obj):
                import numpy as np
                if isinstance(obj, (pd.Series, np.ndarray)):
                    return obj.tolist()
                elif pd.api.types.is_scalar(obj) and pd.isna(obj):
                    return None
                elif isinstance(obj, dict):
                    return {k: convert_nan_to_none(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_nan_to_none(item) for item in obj]
                return obj

            clean_sample = convert_nan_to_none(sample)
            val_keys = [field.name for field in fields(ChatMLSample) if field.name in clean_sample]
            clean_sample = {k: clean_sample[k] for k in val_keys}

            try:
                sample = dacite.from_dict(
                    data_class=ChatMLSample, data=clean_sample, config=dacite.Config(strict=True, check_types=True)
                )
            except Exception as e:
                print(f"Failed to convert to ChatMLSample: {e}")
                print(f"Clean sample: {json.dumps(clean_sample, indent=2)}")
                return None, None, None, None, None, None, None  # 新增text_segment_counts返回

        input_tokens = []
        label_tokens = []
        audio_contents = []  # 仍保留1个完整音频路径（后续外部拆分）
        audio_label_contents = []  # 对应保留1个完整音频路径
        speaker_id = None
        if sample.speaker is not None:
            speaker_id = sample.speaker
        elif sample.misc is not None and "speaker" in sample.misc:
            speaker_id = sample.misc["speaker"]
        text_segment_counts = []  # 新增：记录每一轮audio对应的文本段数（索引对应audio_contents）
        text_segment_counts_for_labels = []

        total_m = len(sample.messages)
        for turn_id, message in enumerate(sample.messages):
            role = message.role
            recipient = message.recipient
            content = message.content
            content_l = []

            # 原有content格式统一逻辑保留
            if isinstance(content, str):
                content_l.append(TextContent(text=content))
            elif isinstance(content, TextContent):
                content_l.append(content)
            elif isinstance(content, AudioContent):
                content_l.append(content)
            elif isinstance(content, list):
                for ele in content:
                    if isinstance(ele, str):
                        content_l.append(TextContent(text=ele))
                    else:
                        content_l.append(ele)

            # 原有前缀处理逻辑保留
            if turn_id == 0:
                prefix = f"<|begin_of_text|><|start_header_id|>{role}<|end_header_id|>\n\n"
            else:
                prefix = f"<|start_header_id|>{role}<|end_header_id|>\n\n"
            prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
            input_tokens.extend(prefix_tokens)
            label_tokens.extend([-100 for _ in prefix_tokens])

            # 原有recipient处理逻辑保留
            if recipient:
                assert role == "assistant", "Recipient is only available for assistant role."
                recipient_tokens = tokenizer.encode(f"{recipient}<|recipient|>", add_special_tokens=False)
                input_tokens.extend(recipient_tokens)
                label_tokens.extend(recipient_tokens)

            # -------------------------- 核心：多轮文本拆分+段数记录（仅处理有text+audio的轮次） --------------------------
            # 标记当前轮是否有对应的text和audio（用于后续段数记录）
            current_turn_has_audio = False
            current_turn_text_segments = []  # 当前轮拆分后的文本段

            if role == "assistant":  # 假设assistant轮是唯一有text+audio的轮次，可根据实际调整
                # 1. 提取当前轮的text和audio（每轮1个text+1个audio）
                turn_text_content = None
                turn_audio_content = None
                for cnt in content_l:
                    if isinstance(cnt, TextContent):
                        turn_text_content = cnt
                    elif isinstance(cnt, AudioContent):
                        turn_audio_content = cnt

                if turn_text_content is not None and turn_audio_content is not None:
                    current_turn_has_audio = True
                    # 2. 拆分当前轮的文本为N段（按text_split_config）
                    full_text_tokens = tokenizer.encode(turn_text_content.text, add_special_tokens=False)
                    if len(full_text_tokens) <= first_text_len:
                        # 文本长度 <= 首段长度：仅1段
                        current_turn_text_segments = [full_text_tokens]
                    else:
                        # 首段取first_text_len，后续每段取subsequent_text_len
                        current_turn_text_segments.append(full_text_tokens[:first_text_len])
                        remaining_text = full_text_tokens[first_text_len:]
                        while len(remaining_text) > 0:
                            take_len = min(subsequent_text_len, len(remaining_text))
                            current_turn_text_segments.append(remaining_text[:take_len])
                            if take_len < len(remaining_text):
                                remaining_text = remaining_text[take_len:]
                            else:
                                remaining_text = []
                                break

                    # 3. 记录当前轮的文本段数（后续audio需拆成该数量）
                    text_segment_counts.append(len(current_turn_text_segments))
                    text_segment_counts_for_labels.append(len(current_turn_text_segments))
                    # 4. 将当前轮的完整audio加入列表（不拆分）
                    audio_contents.append(turn_audio_content)
                    # 5. 处理audio_label_contents（标注逻辑保留）
                    if sample.start_index is None or turn_id >= sample.start_index:
                        audio_label_contents.append(turn_audio_content)
                    else:
                        audio_label_contents.append(None)

            # -------------------------- 文本Token拼接（含多段文本+音频标记） --------------------------
            if current_turn_has_audio:
                # 当前轮有audio：遍历拆分后的文本段，逐段加Token+音频标记
                total_audio_segments = len(current_turn_text_segments)
                for audio_segment_id, text_seg in enumerate(current_turn_text_segments):
                    # 加文本段Token
                    input_tokens.extend(text_seg)
                    # 加标注（assistant轮且在start_index后标注）
                    if sample.start_index is None or turn_id >= sample.start_index:
                        label_tokens.extend(text_seg)
                    else:
                        label_tokens.extend([-100 for _ in text_seg])

                    # 每段文本后加音频标记（<|audio_out_bos|><|AUDIO_OUT|><|audio_out_eos|>）
                    if audio_segment_id != total_audio_segments - 1:
                        audio_mark_tokens = tokenizer.encode(
                            f"<|audio_out_bos|><|AUDIO_OUT|><|audio_out_eos|>",
                            add_special_tokens=False
                        )
                    else:
                        audio_mark_tokens = tokenizer.encode(
                            f"<|audio_out_last_bos|><|AUDIO_OUT|><|audio_out_eos|>",
                            add_special_tokens=False
                        )
                    input_tokens.extend(audio_mark_tokens)
                    # 音频标记标注
                    if sample.start_index is None or turn_id >= sample.start_index:
                        label_tokens.extend(audio_mark_tokens)
                    else:
                        label_tokens.extend([-100 for _ in audio_mark_tokens])
            else:
                # user/system轮无论有无audio / 无audio的assistant）：原有逻辑保留
                for cnt in content_l:
                    if isinstance(cnt, TextContent):
                        text_tokens = tokenizer.encode(cnt.text, add_special_tokens=False)
                        input_tokens.extend(text_tokens)
                        # label_tokens.extend([-100 for _ in text_tokens])
                        if role == "assistant" and (sample.start_index is None or turn_id >= sample.start_index):
                            label_tokens.extend(text_tokens)
                        else:
                            label_tokens.extend([-100 for _ in text_tokens])
                    elif isinstance(cnt, AudioContent):
                        audio_contents.append(content)
                        text_segment_counts.append(1)

                        if role == "user" or role == "system":
                            # Add the text tokens
                            text_tokens = tokenizer.encode(
                                f"<|audio_bos|><|AUDIO|><|audio_eos|>",
                                add_special_tokens=False,
                            )# [128011, 128015, 128012]
                            input_tokens.extend(text_tokens)
                            label_tokens.extend([-100 for _ in text_tokens])
                        elif role == "assistant":
                            print('???\n\n\n\n\n')

            # -------------------------- 原有后缀（eom_id/eot_id）处理保留 --------------------------
            next_id = turn_id + 1
            if role == "assistant" and next_id != total_m and sample.messages[next_id].role == "assistant":
                postfix_tokens = tokenizer.encode("<|eom_id|>", add_special_tokens=False)
            else:
                postfix_tokens = tokenizer.encode("<|eot_id|>", add_special_tokens=False)
            input_tokens.extend(postfix_tokens)
            # 后缀标注
            if role == "assistant" and (sample.start_index is None or turn_id >= sample.start_index):
                label_tokens.extend(postfix_tokens)
            else:
                label_tokens.extend([-100 for _ in postfix_tokens])

        # -------------------------- 返回值：新增text_segment_counts --------------------------
        return (
            input_tokens,          # 原有：所有文本Token
            label_tokens,          # 原有：所有标注Token
            audio_contents,        # 原有：多轮完整音频对象（每轮1个）
            audio_label_contents,  # 原有：多轮完整标注音频对象
            speaker_id,            # 原有：说话人ID
            text_segment_counts,    # 新增：每轮audio对应的文本段数（索引与audio_contents一致）
            text_segment_counts_for_labels,
        )

    except Exception as e:
        print(f"prepare_chatml_sample错误: {str(e)}")
        print(f"样本数据: {json.dumps(sample, indent=2) if isinstance(sample, dict) else str(sample)}")
        return None, None, None, None, None, None


def extract_generation_prompt_from_input_tokens(input_tokens, tokenizer):
    """Extract the generation prompt and reference answer from the input tokens.

    For example:

    Input Text = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n
    What words do you hear from the provided audio? Write it down for me.<|audio_bos|><|AUDIO|><|audio_eos|><|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>\n\nAt first they went by quick, too quick to even get.<|eot_id|>'

    -->

    Prompt = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n
    What words do you hear from the provided audio? Write it down for me.<|audio_bos|><|AUDIO|><|audio_eos|><|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>\n\n',
    Reference = 'At first they went by quick, too quick to even get.'

    Args:
        input_tokens: The input tokens.
        audio_contents: The audio contents.
        tokenizer: The tokenizer to use for decoding the text.

    Returns:
        prompt_tokens: The tokens for the prompt.
        reference_answer: The reference answer.
        num_audios_in_reference: The number of audios in the reference answer.

    """
    input_text = tokenizer.decode(input_tokens)
    generation_prefix = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    postfix = "<|eot_id|>"
    assert generation_prefix in input_text
    generation_prompt_end_loc = input_text.rfind(generation_prefix) + len(generation_prefix)
    generation_prompt = input_text[:generation_prompt_end_loc]
    reference_answer = input_text[generation_prompt_end_loc : input_text.find(postfix, generation_prompt_end_loc)]
    num_audios_in_reference = reference_answer.count(AUDIO_IN_TOKEN) + reference_answer.count(AUDIO_OUT_TOKEN)
    return tokenizer.encode(generation_prompt, add_special_tokens=False), reference_answer, num_audios_in_reference


def prepare_chatml_dataframe_single_process(df, tokenizer):
    """Prepare the ChatML DataFrame."""
    ret = []
    for _, row in df.iterrows():
        result = prepare_chatml_sample(row.to_dict(), tokenizer)
        if result[0] is not None:  # 检查是否成功处理
            input_tokens, label_tokens, audio_contents, audio_label_contents, speaker_id = result
            
            # 处理音频数据
            audio_ids_list = []
            audio_waveforms_list = []
            audio_sample_rates = []
            audio_speaker_indices = []
            audio_ids_start = []
            audio_waveforms_start = []
            label_audio_ids_list = []
            
            current_audio_offset = 0
            current_waveform_offset = 0
            
            for i, (audio_content, audio_label_content) in enumerate(zip(audio_contents, audio_label_contents)):
                # 编码音频为tokens
                audio_tokens = encode_audio_to_tokens(audio_content)
                
                if audio_tokens.numel() > 0:
                    # 记录起始位置
                    audio_ids_start.append(current_audio_offset)
                    
                    # 添加音频tokens
                    audio_ids_list.append(audio_tokens)
                    current_audio_offset += audio_tokens.shape[-1]
                
                # 处理音频标签
                if audio_label_content is not None:
                    label_tokens = encode_audio_to_tokens(audio_label_content)
                    label_audio_ids_list.append(label_tokens)
                
                # 处理音频波形
                if hasattr(audio_content, 'waveform') and audio_content.waveform is not None:
                    waveform = audio_content.waveform
                    if isinstance(waveform, (list, np.ndarray)):
                        waveform = torch.tensor(waveform, dtype=torch.float32)
                    
                    audio_waveforms_start.append(current_waveform_offset)
                    audio_waveforms_list.append(waveform)
                    current_waveform_offset += len(waveform)
                    
                    # 采样率
                    sample_rate = getattr(audio_content, 'sample_rate', 16000)
                    audio_sample_rates.append(sample_rate)
                
                # 说话人索引
                speaker_idx = -1  # 默认未知说话人
                if speaker_id is not None:
                    speaker_idx = hash(speaker_id) % 1000  # 简单的hash映射
                audio_speaker_indices.append(speaker_idx)
            
            # 拼接所有音频数据
            if audio_ids_list:
                audio_ids_concat = torch.cat(audio_ids_list, dim=-1)
            else:
                audio_ids_concat = torch.tensor([[]], dtype=torch.long)
            
            if audio_waveforms_list:
                audio_waveforms_concat = torch.cat(audio_waveforms_list, dim=0)
            else:
                audio_waveforms_concat = torch.tensor([], dtype=torch.float32)
            
            if label_audio_ids_list:
                label_audio_ids = torch.cat(label_audio_ids_list, dim=-1)
            else:
                label_audio_ids = None
            
            # 构建样本
            sample = ChatMLDatasetSample(
                input_ids=torch.tensor(input_tokens, dtype=torch.long),
                label_ids=torch.tensor(label_tokens, dtype=torch.long),
                audio_ids_concat=audio_ids_concat,
                audio_ids_start=torch.tensor(audio_ids_start, dtype=torch.long),
                audio_waveforms_concat=audio_waveforms_concat,
                audio_waveforms_start=torch.tensor(audio_waveforms_start, dtype=torch.long),
                audio_sample_rate=torch.tensor(audio_sample_rates, dtype=torch.float32),
                audio_speaker_indices=torch.tensor(audio_speaker_indices, dtype=torch.long),
                label_audio_ids=label_audio_ids,  # 设置音频标签
            )
            ret.append(sample)
        else:
            logger.warning(f"Failed to process sample at index {_}")
    return ret


def prepare_chatml_dataframe(df, tokenizer, num_process=16):
    if num_process is None:
        return prepare_chatml_dataframe_single_process(df, tokenizer)
    else:
        num_process = max(min(len(df) // 1000, num_process), 1)
        workloads = np.array_split(df, num_process)
        with mp.Pool(num_process) as pool:
            ret = pool.starmap(
                prepare_chatml_dataframe_single_process, [(workload, tokenizer) for workload in workloads]
            )
    return sum(ret, [])


class DatasetInterface(ABC):
    @abstractmethod
    def __getitem__(self, idx) -> Union["ChatMLDatasetSample", "RankedChatMLDatasetSampleTuple"]:
        """Retrieve a dataset sample by index."""
        raise NotImplementedError


class IterableDatasetInterface(ABC):
    @abstractmethod
    def __iter__(self) -> Union["ChatMLDatasetSample", "RankedChatMLDatasetSampleTuple"]:
        """Retrieve a sample by iterating through the dataset."""
        raise NotImplementedError


@dataclass
class DatasetInfo:
    dataset_type: str
    group_type: Optional[str] = None
    mask_text: Optional[bool] = None  # Whether to mask the text tokens for pretraining samples.
