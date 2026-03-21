"""Example for using HiggsAudio for generating both the transcript and audio in an interleaved manner."""

import os
import json

from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse, AsyncHiggsAudioStreamer
import torch
import torchaudio
import numpy as np
import time
from loguru import logger
import click

# edit accelerate
from accelerate import Accelerator
accelerator = Accelerator()
from accelerate.utils import gather_object

from boson_multimodal.model.higgs_audio.modeling_higgs_audio import HiggsAudioEncoder, AudioAligner

from boson_multimodal.data_types import ChatMLSample, Message, AudioContent

from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern

import asyncio
from threading import Thread

MODEL_DICT = {
    "0915_llama_aligner": {  # 56-57 + 74 llama and aligner only tune aligner
        "MODEL_ID": "0915_without_audio_in_token_llaso",
        "STEP": 35781,  # 35781
    },
    "0916_llama_proj": {  # 40000 56-57 + 74 llama and proj only tune proj
        "MODEL_ID": "0916_without_audio_in_token_llama",
        "STEP": 40000,  #
    },
    "0916_llama_proj_finetune": {  # 32000 63.67 + 67 llama and proj tune llama and proj
        "MODEL_ID": "0916_without_audio_in_token_llama_finetune",
        "STEP": 32000,  #
    },
    "0916_llama_aligner": {  # llama and aligner only tune aligner
        # 53.67 + 73 (68000)
        # 54.33 + 73 (81795)
        "MODEL_ID": "0916_without_audio_in_token_llama_aligner",
        "STEP": 81795,  #
    },
    "0916_llama_aligner_finetune": {  # llama and aligner tune llama and aligner
        # 64.33 + 68.33 (54000)
        # 65.33 + 69 (81795)
        "MODEL_ID": "0916_without_audio_in_token_llama_aligner_finetune",
        "STEP": 81795,  #
    },
    "0916_llaso": {  # finish 54-55 + 66 llaso and aligner only tune aligner
        "MODEL_ID": "0916_without_audio_in_token_llaso_instruct",
        "STEP": 40899,  # 40899
    },
    "0916_llaso_finetune": {  # llaso and aligner tune llaso and aligner
        # 62 + 68 (32000) 
        # 65 + 68 (54000)
        # 63.33 + 67 (81795)
        "MODEL_ID": "0916_without_audio_in_token_llaso_finetune",
        "STEP": 81795,  # 
    },
    "0916_llaso_tts": {  # llama and proj only tune tokenizer and audioffn
        "MODEL_ID": "0916_without_audio_in_token_llaso_tts",
        "STEP": 40899,  #
    },
    "0918_higgs": {
        "MODEL_ID": "0918_without_audio_in_token_higgs_tts",
        "STEP": 11027,  #
    },
    "0919_understanding2tts_add_audio_attn": {
        # 5000 61.33 + 64.67
        "MODEL_ID": "0919_without_audio_in_token_llaso_tts_with_audio_attention",
        "STEP": 45207,  #
    },
    "0919_understanding2tts_finetune": {
        # 7000 56.67 + 62.67
        "MODEL_ID": "0919_without_audio_in_token_llaso_tts_finetune",
        "STEP": 59000,  #
    },
    "1001_lr3e4": {
        "MODEL_ID": "1001_lr3e4",
        "STEP": 55000,  #
    },
    "1003_lr3e4_emilia": {
        "MODEL_ID": "1003_lr3e4_emilia",
        "STEP": 90000,  #
    },
    "1005_lr3e4_emilia_freeze_added_tokens": {
        "MODEL_ID": "1005_lr3e4_emilia_freeze_added_tokens",
        "STEP": 207000,  #
    },
    "pretrain2sft": {
        "MODEL_ID": "1012_save_pretrain2sft",
        "STEP": 55000,  #
    },
    "pretrain2sft_3epoch": {
        "MODEL_ID": "1010_pretrain2sft",
        "STEP": 81795,  #
    },
    "1021_attn": {
        "MODEL_ID": "1021_attn_32000_tts",
        "STEP": 59000,  #
    },
    "1021_attn_emilia": {
        "MODEL_ID": "1021_attn_81795_tts_emilia",
        "STEP": 386000,
    },
    "1021_attn_81795": {
        "MODEL_ID": "1021_attn_81795_tts",
        "STEP": 81795,
    },
    "1028_attn_last_out_28000": {
        "MODEL_ID": "1028_attn_last_out_28000_tts",
        "STEP": 20000,
    },
    "1028_attn_last_out_81795": {
        "MODEL_ID": "1028_attn_last_out_81795_tts",
        "STEP": 81795,
    },
    "1028_attn_last_out_81795_tts_emilia": {
        "MODEL_ID": "1028_attn_last_out_81795_tts_emilia",
        "STEP": 135000,
    },
    "1107_ds_1_32_64_10_10_40_40_tts": {
        "MODEL_ID": "1107_ds_1_32_64_10_10_40_40_tts",
        "STEP": 21000,
    },
    "1107_ds_1_32_64_10_10_40_40_tts_emilia": {
        "MODEL_ID": "1107_ds_1_32_64_10_10_40_40_tts_emilia",
        "STEP": 11000,
    },
    "1113_ds_1_32_56_10_10_40_40_14000_tts_without_ultrachat": {
        "MODEL_ID": "1113_ds_1_32_56_10_10_40_40_14000_tts_without_ultrachat",
        "STEP": 7000,
    },
    "1113_ds_1_32_56_10_10_40_40_20691_tts_without_ultrachat": {
        "MODEL_ID": "1113_ds_1_32_56_10_10_40_40_20691_tts_without_ultrachat",
        "STEP": 5000,
    },
    "1115_1_30_30_10_10_40_40_tts": {
        "MODEL_ID": "1115_1_30_30_10_10_40_40_tts",
        "STEP": 34000,
    },
    "1115_09_30_30_10_10_40_40_tts": {
        "MODEL_ID": "1115_09_30_30_10_10_40_40_tts",
        "STEP": 47478,
    },
    "1115_1_30_30_10_10_40_40_tts_emilia_no_ds": {
        "MODEL_ID": "1115_1_30_30_10_10_40_40_tts_emilia_no_ds",
        "STEP": 215000,
    },
    "paper_model": {
        "MODEL_ID": "paper_model",
        "STEP": 320000,
    },
    "test_emo": {  # ac + emilia (> 0.05, 125000)
        "MODEL_ID": "step_3_1128",
        "STEP": 9000,
    },
    "test_emo_1203": {  # ac + emilia (> 0.05, 320000)
        "MODEL_ID": "step_3_1203",
        "STEP": 11510,
    },
    "test_emo_1204": {  # ac + emilia + emo (> 0.05, 320000)
        "MODEL_ID": "step_3_1204",
        "STEP": 8950,
    },
    "test_emo_1205": {  # ac + emilia + EmoVoice + emo (> 0.05, 320000)
        "MODEL_ID": "step_3_1205",
        "STEP": 4000,
    },
    "test_emo_1205_1": {  # ac + emilia + EmoVoice(a+t) + emo (> 0.05, 320000)
        "MODEL_ID": "step_3_1205_1",
        "STEP": 26421,
    },
    "test_emo_1206": {  # ac + more emilia + EmoVoice(a+t) + emo (> 0.05, 320000)
        "MODEL_ID": "step_3_1206",
        "STEP": 58879,
    },
    "test_emo_1208": {  # ac + more emilia + EmoVoice(a+t) + emo (> 0.05, 320000)
        "MODEL_ID": "step_3_1208",
        "STEP": 46504,
    },
    "test_emo_1212": {  # EmoVoice(a*2+t) + libritts (> 0.05, 320000)
        "MODEL_ID": "step_4_1212",
        "STEP": 32000,
    },
    "test_emo_1213": {  # EmoVoice(a+t) + libritts(0.8a+0.2t) + two merge (> 0.05, 320000)
        "MODEL_ID": "step_4_1213",
        "STEP": 50004,
    },
    "test_3_1216": {  # EmoVoice(a+t) + libritts_merge + VCTK (> 0.05, 320000)
        "MODEL_ID": "step_3_1216",
        "STEP": 73672,
    },
    "test_3_1219": {  # ac + ac_emo + EmoVoice(a) (> 0.05, 320000)
        "MODEL_ID": "step_3_1219",
        "STEP": 20000,
    },
    "test_3_1220": {  # ac (> 0.05, 320000)
        "MODEL_ID": "step_3_1220",
        "STEP": 13000,
    },
    "test_4_1221": {  # ac_emo + EmoVoice(a) (test_3_1220)
        "MODEL_ID": "step_4_1221",
        "STEP": 5993,
    },
    "test_4_1221_1": {  # ac_emo + EmoVoice(a) (test_3_1220)
        "MODEL_ID": "step_4_1221_1",
        "STEP": 7000,
    },
    "test_4_1221_2": {  # ac_emo (filter) + EmoVoice(a) (test_3_1220)
        "MODEL_ID": "step_4_1221_2",
        "STEP": 6000,
    },
    "test_4_1221_3": {  # ac (filter) + EmoVoice(a) (test_3_1220)
        "MODEL_ID": "step_4_1221_3",
        "STEP": 9210,
    },
    "test_4_1222": {  # ac_emo (filter correct) + EmoVoice(a) (test_3_1220)
        "MODEL_ID": "step_4_1222",
        "STEP": 11000,
    },
    "test_4_1222_2": {  # acx2 (filter correct) + EmoVoice(a) (test_3_1220)
        "MODEL_ID": "step_4_1222_2",
        "STEP": 2000,
    },
    "test_4_1224": {  # acx2 (filter correct) + EmoVoice(a) + EmoVoice (test_3_1220)
        "MODEL_ID": "step_4_1224",
        "STEP": 4000,
    },
}

DATE = "test_4_1224"
MODEL_ID = MODEL_DICT[DATE]["MODEL_ID"]
STEP = MODEL_DICT[DATE]["STEP"]

if "llaso" in DATE:
    use_llaso_architecture = True
else:
    use_llaso_architecture = False
load_llaso_init = False

MODEL_PATH = f"/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/train-higgs-audio/output_train/{MODEL_ID}/checkpoint-{STEP}"
AUDIO_TOKENIZER_PATH = "./higgs-audio-v2-tokenizer"

def get_chat_ml(request_type="a", request="", response_type="ta"):
    # system_prompt = (
    #     "Generate audio following instruction.\n\n"
    #     "<|scene_desc_start|>\n"
    #     "Audio is recorded from a quiet room.\n"
    #     "<|scene_desc_end|>"
    # )

    system_prompt = (
        "Answer the question of the user!"
    )

    if request_type == "a":
        content = AudioContent(audio_url=request)
    else:
        content = request

    messages = [
        Message(
            role="system",
            content=system_prompt,
        ),
        Message(
            role="user",
            content=content,
            response_type=response_type,
        ),
    ]
    chat_ml_sample = ChatMLSample(messages=messages)
    return chat_ml_sample

# 异步处理函数：实时解析token、分割片段、合成音频（8级RVQ实时音频片段合成）
async def process_single_stream(streamer, text_stream_path, audio_segments_path, 
                              accumulated_text, accumulated_audio_segments, serve_engine, cache_codes_length=10, code2wavnum=960):
    assert cache_codes_length % 2 == 0

    AUDIO_BOS_TOKEN = 1024  # 8级RVQ共用BOS值
    AUDIO_EOS_TOKEN = 1025  # 8级RVQ共用EOS值
    NUM_CODEBOOKS = 8  # 固定8级RVQ

    current_audio_tokens = []  # 缓存当前片段的token（形状：[seq_len, 8]）
    in_audio_segment = False   # 是否处于音频片段中

    cache_audio_tokens = None
    last_audio = False

    time_after_decode = time.time()
    cur_token_time = time.time()
    async for delta in streamer:
        if int(accelerator.process_index) == 2:
            logger.debug(f"Process {accelerator.process_index}: Token Between Time: {time.time() - cur_token_time}, {delta}")
        cur_token_time = time.time()

        # 处理文本token
        if delta.text is not None:
            if delta.text == '<|audio_out_last_bos|>':
                last_audio = True
            if delta.text != '<|audio_out_bos|>' and delta.text != '<|AUDIO_OUT|>' and delta.text != '<|audio_out_eos|>' and delta.text != '<|audio_out_last_bos|>' and delta.text != '<|eot_id|>':
                accumulated_text[0] += delta.text  # 关键：修改外部传入的列表元素
            if int(accelerator.process_index) == 2:
                logger.debug(f"Process {accelerator.process_index}: Text stream: {delta.text}")  # 用debug级别避免日志刷屏

        # 处理音频token（8级RVQ专用逻辑）
        if delta.audio_tokens is not None:
            audio_tokens = delta.audio_tokens.cpu().numpy()  # 形状：[8, token_len]
            if len(audio_tokens.shape) == 1:
                num_codebooks = 8
                token_len = 1
            else:
                num_codebooks, token_len = audio_tokens.shape
            assert num_codebooks == NUM_CODEBOOKS, f"Expected 8 codebooks, got {num_codebooks}"

            # 逐时间步解析
            for t in range(token_len):
                if len(audio_tokens.shape) == 1:
                    step_tokens = audio_tokens
                else:
                    step_tokens = audio_tokens[:, t]  # 当前时间步的8个码本token

                # 检测8码本同时为BOS：片段开始
                if not in_audio_segment and (step_tokens == AUDIO_BOS_TOKEN).all():
                    in_audio_segment = True
                    current_audio_tokens = []
                    logger.info(f"Detected 8-codebook BOS (Process {accelerator.process_index})")
                    continue

                # 检测8码本同时为EOS：片段结束
                if in_audio_segment and (step_tokens == AUDIO_EOS_TOKEN).all():
                    in_audio_segment = False
                    # 转换为[8, seq_len] tensor并处理delay
                    segment_tensor = torch.tensor(current_audio_tokens, device=serve_engine.device).transpose(0, 1).contiguous()
                    segment_tensor = revert_delay_pattern(segment_tensor).clip(0, serve_engine.audio_codebook_size - 1)

                    if cache_audio_tokens is not None and cache_codes_length != 0:
                        segment_tensor = torch.cat((cache_audio_tokens, segment_tensor), dim = 1)

                    time_before_decode = time.time()
                    # 实时解码音频片段
                    audio_waveform = serve_engine.audio_tokenizer.decode(segment_tensor.unsqueeze(0))[0, 0]

                    if cache_audio_tokens is not None and cache_codes_length != 0:
                        audio_waveform = audio_waveform[cache_codes_length * code2wavnum // 2:]
                    if not last_audio and cache_codes_length != 0:
                        audio_waveform = audio_waveform[:-cache_codes_length * code2wavnum // 2]

                    accumulated_audio_segments.append(audio_waveform)
                    # 保存片段
                    seg_idx = len(accumulated_audio_segments) - 1
                    if cache_codes_length != 0:
                        cache_audio_tokens = segment_tensor[..., -cache_codes_length:]

                    cur_time = time.time()
                    if int(accelerator.process_index) == 2:
                        logger.debug(f"Process {accelerator.process_index}: Decode Time: {cur_time - time_before_decode}")
                        logger.debug(f"Process {accelerator.process_index}: Time Between Two wavs: {cur_time - time_after_decode}")
                        logger.debug(f"Process {accelerator.process_index}: Wave time {audio_waveform.shape[-1] / 24000}")
                    time_after_decode = time.time()

                    # np.save(f"{audio_segments_path}_seg_{seg_idx}.npy", audio_waveform)
                    # logger.info(f"Process {accelerator.process_index}: Saved audio segment {seg_idx}")
                    continue

                # 缓存音频token（仅在片段中）
                if in_audio_segment:
                    current_audio_tokens.append(step_tokens)  # 形状：[seq_len, 8]

    # 实时写入文本流（上下文管理器确保关闭）
    if accumulated_text[0]:
        with open(text_stream_path, "w", encoding="utf-8") as f:
            f.write(accumulated_text[0])

    # 拼接所有实时片段为完整音频
    if accumulated_audio_segments:
        final_audio = np.concatenate(accumulated_audio_segments, axis=0)
        logger.info(f"Process {accelerator.process_index}: Final audio length {len(final_audio)} samples")
        return final_audio

    return None

# 异步处理函数：实时解析token、分割片段、合成音频（8级RVQ实时音频片段合成）
async def process_single_stream_solve_last(streamer, text_stream_path, audio_segments_path, 
                              accumulated_text, accumulated_audio_segments, serve_engine, cache_codes_length=10, code2wavnum=960):
    assert cache_codes_length % 2 == 0

    AUDIO_BOS_TOKEN = 1024  # 8级RVQ共用BOS值
    AUDIO_EOS_TOKEN = 1025  # 8级RVQ共用EOS值
    NUM_CODEBOOKS = 8  # 固定8级RVQ

    current_audio_tokens = []  # 缓存当前片段的token（形状：[seq_len, 8]）
    in_audio_segment = False   # 是否处于音频片段中

    cache_audio_tokens = None
    last_audio = False

    time_after_decode = time.time()
    cur_token_time = time.time()
    async for delta in streamer:
        if int(accelerator.process_index) == 2:
            logger.debug(f"Process {accelerator.process_index}: Token Between Time: {time.time() - cur_token_time}, {delta}")
        cur_token_time = time.time()

        # 处理文本token
        if delta.text is not None:
            if delta.text == '<|audio_out_last_bos|>':
                last_audio = True
            if delta.text != '<|audio_out_bos|>' and delta.text != '<|AUDIO_OUT|>' and delta.text != '<|audio_out_eos|>' and delta.text != '<|audio_out_last_bos|>' and delta.text != '<|eot_id|>':
                accumulated_text[0] += delta.text  # 关键：修改外部传入的列表元素
            if int(accelerator.process_index) == 2:
                logger.debug(f"Process {accelerator.process_index}: Text stream: {delta.text}")  # 用debug级别避免日志刷屏

        # 处理音频token（8级RVQ专用逻辑）
        if delta.audio_tokens is not None:
            audio_tokens = delta.audio_tokens.cpu().numpy()  # 形状：[8, token_len]
            if len(audio_tokens.shape) == 1:
                num_codebooks = 8
                token_len = 1
            else:
                num_codebooks, token_len = audio_tokens.shape
            assert num_codebooks == NUM_CODEBOOKS, f"Expected 8 codebooks, got {num_codebooks}"

            # 逐时间步解析
            for t in range(token_len):
                if len(audio_tokens.shape) == 1:
                    step_tokens = audio_tokens
                else:
                    step_tokens = audio_tokens[:, t]  # 当前时间步的8个码本token

                # 检测8码本同时为BOS：片段开始
                if not in_audio_segment and (step_tokens == AUDIO_BOS_TOKEN).all():
                    in_audio_segment = True
                    current_audio_tokens = []
                    if int(accelerator.process_index) == 2:
                        logger.info(f"Detected 8-codebook BOS (Process {accelerator.process_index})")
                    continue

                # 检测8码本同时为EOS：片段结束
                if in_audio_segment and (step_tokens == AUDIO_EOS_TOKEN).all():
                    in_audio_segment = False
                    # 转换为[8, seq_len] tensor并处理delay
                    segment_tensor = torch.tensor(current_audio_tokens, device=serve_engine.device).transpose(0, 1).contiguous()
                    segment_tensor = revert_delay_pattern(segment_tensor).clip(0, serve_engine.audio_codebook_size - 1)

                    if cache_audio_tokens is not None and cache_codes_length != 0:
                        segment_tensor = torch.cat((cache_audio_tokens, segment_tensor), dim = 1)

                    time_before_decode = time.time()
                    # 实时解码音频片段
                    audio_waveform = serve_engine.audio_tokenizer.decode(segment_tensor.unsqueeze(0))[0, 0]

                    if cache_audio_tokens is not None and cache_codes_length != 0:
                        audio_waveform = audio_waveform[cache_codes_length * code2wavnum // 2:]
                    if not last_audio and cache_codes_length != 0:
                        audio_waveform = audio_waveform[:-cache_codes_length * code2wavnum // 2]

                    accumulated_audio_segments.append(audio_waveform)
                    # 保存片段
                    seg_idx = len(accumulated_audio_segments) - 1
                    if cache_codes_length != 0:
                        cache_audio_tokens = segment_tensor[..., -cache_codes_length:]

                    cur_time = time.time()
                    if int(accelerator.process_index) == 2:
                        logger.debug(f"Process {accelerator.process_index}: Decode Time: {cur_time - time_before_decode}")
                        logger.debug(f"Process {accelerator.process_index}: Time Between Two wavs: {cur_time - time_after_decode}")
                        logger.debug(f"Process {accelerator.process_index}: Wave time {audio_waveform.shape[-1] / 24000}")
                    time_after_decode = time.time()

                    # np.save(f"{audio_segments_path}_seg_{seg_idx}.npy", audio_waveform)
                    # logger.info(f"Process {accelerator.process_index}: Saved audio segment {seg_idx}")
                    continue
                
                elif in_audio_segment and last_audio and len(current_audio_tokens) >= 47:
                    segment_tensor = torch.tensor(current_audio_tokens, device=serve_engine.device).transpose(0, 1).contiguous()
                    segment_tensor = revert_delay_pattern(segment_tensor).clip(0, serve_engine.audio_codebook_size - 1)
                    if cache_audio_tokens is not None and cache_codes_length != 0:
                        segment_tensor = torch.cat((cache_audio_tokens, segment_tensor), dim = 1)

                    time_before_decode = time.time()
                    # 实时解码音频片段
                    audio_waveform = serve_engine.audio_tokenizer.decode(segment_tensor.unsqueeze(0))[0, 0]

                    if cache_audio_tokens is not None and cache_codes_length != 0:
                        audio_waveform = audio_waveform[cache_codes_length * code2wavnum // 2:]
                    if cache_codes_length != 0:
                        audio_waveform = audio_waveform[:-cache_codes_length * code2wavnum // 2]

                    accumulated_audio_segments.append(audio_waveform)
                    # 保存片段
                    seg_idx = len(accumulated_audio_segments) - 1
                    if cache_codes_length != 0:
                        cache_audio_tokens = segment_tensor[..., -cache_codes_length:]

                    cur_time = time.time()
                    if int(accelerator.process_index) == 2:
                        logger.debug(f"Process {accelerator.process_index}: Decode Time: {cur_time - time_before_decode}")
                        logger.debug(f"Process {accelerator.process_index}: Time Between Two wavs: {cur_time - time_after_decode}")
                        logger.debug(f"Process {accelerator.process_index}: Wave time {audio_waveform.shape[-1] / 24000}")
                    time_after_decode = time.time()

                    current_audio_tokens = current_audio_tokens[-7:]

                # 缓存音频token（仅在片段中）
                if in_audio_segment:
                    current_audio_tokens.append(step_tokens)  # 形状：[seq_len, 8]

    # 实时写入文本流（上下文管理器确保关闭）
    if accumulated_text[0]:
        with open(text_stream_path, "w", encoding="utf-8") as f:
            f.write(accumulated_text[0])

    # 拼接所有实时片段为完整音频
    if accumulated_audio_segments:
        final_audio = np.concatenate(accumulated_audio_segments, axis=0)
        logger.info(f"Process {accelerator.process_index}: Final audio length {len(final_audio)} samples")
        return final_audio

    return None

def main(dataset_dir, audio_path, json_path, dataset_flag, request_type):
    audio_path = os.path.join(dataset_dir, audio_path)
    json_path = os.path.join(dataset_dir, json_path)

    dataset_info = []
    with open(json_path, 'r') as f:
        dataset_info = json.load(f)

    requests = []
    correct_responses = []
    audio_ids = []
    questions = []

    for cur_audio in dataset_info:
        if request_type == "a":
            requests.append(os.path.join(audio_path, cur_audio['audio_path']))
        else:
            requests.append(cur_audio['question'])
        correct_responses.append(cur_audio['answer'])
        audio_id, ext = os.path.splitext(cur_audio['audio_path'])
        audio_ids.append(audio_id)
        questions.append(cur_audio['question'])

    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # edit accelerate
    device_id = accelerator.local_process_index
    device = f"cuda:{device_id}"


    logger.info(f"Using device: {device}")

    serve_engine = HiggsAudioServeEngine(
        MODEL_PATH,
        AUDIO_TOKENIZER_PATH,
        device=device,
    )

    # print(serve_engine.model, 'see model')

    if load_llaso_init:
        serve_engine.model.audio_encoder_proj = AudioAligner().to(device, dtype=torch.bfloat16)
        from safetensors import safe_open
        file_path = "LLaSO-Base-3.8B-Instruct/model-00002-of-00002.safetensors"
        higgs_params = dict(serve_engine.model.named_parameters())
        # 打开 Safetensors 文件
        with safe_open(file_path, framework="pt", device="cuda:0") as f:
            # 获取所有张量名称
            for cur_name in f.keys():
                if 'mm_audio_aligner' not in cur_name:
                    continue
                # model.mm_audio_aligner.projector.0.weight
                param = f.get_tensor(cur_name)
                print(param, cur_name)
                cur_id = '.'.join(cur_name.split('.')[2:])
                higgs_params[f'audio_encoder_proj.{cur_id}'].data.copy_(param)
                print(f'finish loading audio_encoder_proj.{cur_id}')

    logger.info("Starting generation...")
    start_time = time.time()

    # 关键：创建主线程的事件循环（用于streamer复用）
    main_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(main_loop)

    # 每个进程（卡）初始化自己的streamer：传入主线程的循环
    streamer = AsyncHiggsAudioStreamer(
        tokenizer=serve_engine.tokenizer,
        skip_prompt=True,
        audio_num_codebooks=serve_engine.audio_num_codebooks,
        decode_kwargs={"skip_special_tokens": True},
        loop=main_loop,  # 传入主线程循环，子线程复用
    )

    batch_size = 8
    responses = []
    audio_save_paths = []
    result_save_dir = os.path.join("generations", MODEL_ID, f"{STEP}_steps", dataset_flag)
    result_json_save_path = os.path.join(result_save_dir, f"{request_type}_as_input.jsonl")
    audio_save_dir = os.path.join(result_save_dir, f"{request_type}_as_input")
    try:
        os.makedirs(audio_save_dir)
    except Exception as e:
        pass

    f_json = open(result_json_save_path, "w", encoding="utf-8")



    cold_request = requests[-1]
    cold_audio_id = audio_ids[-1]
    cold_question = questions[-1]
    cold_correct = correct_responses[-1]
    # 构建当前批次的chat_ml样本
    cold_sample = get_chat_ml(request_type=request_type, request=cold_request, response_type="ta")
    # 当前样本的保存路径
    input_text_path = os.path.join(audio_save_dir, f"{cold_audio_id}_in.txt")
    text_stream_path = os.path.join(audio_save_dir, f"{cold_audio_id}_out.txt")
    audio_segments_prefix = os.path.join(audio_save_dir, f"{cold_audio_id}_audio_segment")
    final_audio_path = os.path.join(audio_save_dir, f"{cold_audio_id}.wav")
    # 初始化积累变量
    accumulated_text = [""]  # 列表存储，确保内部修改能被外部访问
    accumulated_audio_segments = []

    serve_engine.generate(
        chat_ml_sample=cold_sample,
        max_new_tokens=2048,
        temperature=1.0,
        top_p=0.5,
        top_k=1,
        stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        streamer=None,
    )

    


    for batch_start in range(0, len(audio_ids), batch_size):
        batch_end = min(batch_start + batch_size, len(audio_ids))
    # for batch_start in range(0, 3 * batch_size, batch_size):
    #     batch_end = min(batch_start + batch_size, 3 * batch_size)

        batch_requests = requests[batch_start:batch_end]
        batch_audio_ids = audio_ids[batch_start:batch_end]
        batch_questions = questions[batch_start:batch_end]
        batch_correct = correct_responses[batch_start:batch_end]

        logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(audio_ids)-1)//batch_size + 1}")

        # 构建当前批次的chat_ml样本
        batch_samples = [
            get_chat_ml(request_type=request_type, request=req, response_type="ta")
            for req in batch_requests
        ]

        # 多卡分配样本（每张卡处理1个样本）
        with accelerator.split_between_processes(batch_samples) as process_samples:
            # 处理样本分配不均：若当前进程无样本，构造空结果（必须参与通信，不能None）
            if len(batch_audio_ids[accelerator.process_index::accelerator.num_processes]) < 1:
                process_result = [None]
            else:
                # 单卡仅处理1个样本
                sample = process_samples[0]
                # 获取当前样本的元数据（按进程索引取对应元素）
                audio_id = batch_audio_ids[accelerator.process_index::accelerator.num_processes][0]
                question = batch_questions[accelerator.process_index::accelerator.num_processes][0]
                correct_resp = batch_correct[accelerator.process_index::accelerator.num_processes][0]

                # 当前样本的保存路径
                input_text_path = os.path.join(audio_save_dir, f"{audio_id}_in.txt")
                with open(input_text_path, 'w', encoding='utf-8') as f:
                    f.write(question)
                text_stream_path = os.path.join(audio_save_dir, f"{audio_id}_out.txt")
                audio_segments_prefix = os.path.join(audio_save_dir, f"{audio_id}_audio_segment")
                final_audio_path = os.path.join(audio_save_dir, f"{audio_id}.wav")

                # 初始化积累变量
                accumulated_text = [""]  # 列表存储，确保内部修改能被外部访问
                accumulated_audio_segments = []

                # 定义生成任务（在子线程中运行，避免阻塞异步流）
                def generate_task():
                    serve_engine.generate(
                        chat_ml_sample=sample,
                        max_new_tokens=2048,
                        temperature=1.0,
                        top_p=0.5,
                        top_k=1,
                        stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                        streamer=streamer,  # 传入实时流处理器
                    )

                # 启动生成线程
                thread = Thread(target=generate_task)
                thread.start()

                # 异步处理实时流（8级RVQ音频片段合成）
                # 注意：这里直接使用前面创建的 main_loop，不再新建循环
                final_audio = None
                try:
                    final_audio = main_loop.run_until_complete(
                        process_single_stream_solve_last(
                            streamer=streamer,
                            text_stream_path=text_stream_path,
                            audio_segments_path=audio_segments_prefix,
                            accumulated_text=accumulated_text,
                            accumulated_audio_segments=accumulated_audio_segments,
                            serve_engine=serve_engine
                        )
                    )
                finally:
                    thread.join()  # 等待生成线程结束，无需关闭循环（后续批次复用）

                # 保存最终音频（实时片段拼接结果）
                if final_audio is not None:
                    torchaudio.save(
                        final_audio_path,
                        torch.from_numpy(final_audio)[None, :],  # 形状：[1, samples]
                        serve_engine.audio_tokenizer.sampling_rate
                    )
                    logger.info(f"Process {accelerator.process_index}: Saved final audio to {final_audio_path}")

                # 当前样本的结果字典
                process_result = [{
                    "audio_id": audio_id,
                    "response": accumulated_text[0],  # 取列表第0个元素
                    "correct_response": correct_resp,
                    "path": final_audio_path
                }]

        # 关键：在split上下文之外等待所有进程，再收集结果
        accelerator.wait_for_everyone()  # 确保所有进程都完成当前批次处理
        all_results = gather_object(process_result)  # 此时所有进程都已准备好
        # print(all_results, 'all_results???')

        # 主进程写入最终JSONL（避免多进程写冲突）
        if accelerator.is_main_process:
            for res in all_results:
                if res is not None:
                    # print('res???', res)
                    f_json.write(json.dumps(res, ensure_ascii=True) + "\n")

        logger.info(f"Batch {batch_start//batch_size + 1} completed (Total time: {time.time()-start_time:.2f}s)")

    # 所有批次完成后，关闭主线程循环
    main_loop.close()
    f_json.close()
    logger.info(f"All generations completed. Total time: {time.time()-start_time:.2f}s")

if __name__ == "__main__":
    # dataset_flag = "llama-questions"
    # dataset_flag = "triviaqa"
    # dataset_flag = "web-questions"
    # dataset_flag = "alpacaeval"
    dataset_flag = "commoneval"
    # dataset_flag = "ifeval"
    # dataset_flag = "mmsu"
    # dataset_flag = "openbookqa"
    # dataset_flag = "wildvoice"
    # dataset_flag = "UnderEmotion-en"
    # dataset_flag = "GenEmotion-en"
    dataset_dir = f"VoiceBench/processed/{dataset_flag}"
    audio_path = "audio"
    json_path = f"{dataset_flag}.json"
    request_type = "a"
    
    main(dataset_dir, audio_path, json_path, dataset_flag, request_type)
