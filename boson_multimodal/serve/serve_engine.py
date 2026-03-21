import asyncio
import base64
import torch
import torchaudio
import numpy as np
from io import BytesIO
from dataclasses import dataclass
from typing import List, Optional, Union
from copy import deepcopy
from transformers import AutoTokenizer, AutoProcessor
from transformers.cache_utils import StaticCache
from transformers.generation.streamers import BaseStreamer
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from dataclasses import asdict
from loguru import logger
import threading
import librosa
import time

from ..dataset.chatml_dataset import ChatMLSample, ChatMLDatasetSample, prepare_chatml_sample
from ..model.higgs_audio import HiggsAudioModel, HiggsAudioConfig
from ..model.higgs_audio.utils import revert_delay_pattern
from ..data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from ..audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer

import torch.cuda.nvtx as nvtx


@dataclass
class HiggsAudioStreamerDelta:
    """Represents a chunk of generated content, either text or audio tokens."""

    text: Optional[str] = None
    text_tokens: Optional[torch.Tensor] = None
    audio_tokens: Optional[torch.Tensor] = None
    finish_reason: Optional[str] = None


class AsyncHiggsAudioStreamer(BaseStreamer):
    """
    Async streamer that handles both text and audio token generation from Higgs-Audio model.
    Stores chunks in a queue to be consumed by downstream applications.

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenizer used to decode text tokens.
        skip_prompt (`bool`, *optional*, defaults to `False`):
            Whether to skip the prompt tokens in generation.
        timeout (`float`, *optional*):
            The timeout for the queue. If `None`, the queue will block indefinitely.
        decode_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the tokenizer's `decode` method.

    Examples:
        ```python
        >>> from transformers import AutoTokenizer
        >>> from threading import Thread
        >>> import asyncio

        >>> tokenizer = AutoTokenizer.from_pretrained("path/to/higgs/tokenizer")
        >>> model = HiggsAudioModel.from_pretrained("path/to/higgs/model")
        >>> inputs = tokenizer(["Generate some text and audio:"], return_tensors="pt")

        >>> async def main():
        ...     streamer = AsyncHiggsAudioStreamer(tokenizer)
        ...     generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=20)
        ...     thread = Thread(target=model.generate, kwargs=generation_kwargs)
        ...     thread.start()
        ...
        ...     async for delta in streamer:
        ...         if delta.text is not None:
        ...             print("Text:", delta.text)
        ...         if delta.audio_tokens is not None:
        ...             print("Audio tokens shape:", delta.audio_tokens.shape)
        >>> asyncio.run(main())
        ```
    """

    def __init__(
        self,
        tokenizer: "AutoTokenizer",
        skip_prompt: bool = False,
        timeout: Optional[float] = None,
        audio_num_codebooks: int = 1,
        loop: asyncio.AbstractEventLoop = None,  # 新增：接收主线程的循环
        **decode_kwargs,
    ):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.timeout = timeout
        self.decode_kwargs = decode_kwargs
        self.audio_num_codebooks = audio_num_codebooks
        self.queue = asyncio.Queue()
        self.stop_signal = None

        # 关键：使用传入的主线程循环，或获取当前运行的循环（确保是主线程的）
        self.loop = loop or asyncio.get_running_loop()
        self.has_asyncio_timeout = hasattr(asyncio, "timeout")
        self.next_tokens_are_prompt = True

    def put(self, value: torch.Tensor):
        """Receives tokens and processes them as either text or audio tokens."""
        if value.shape[0] > 1 and not self.next_tokens_are_prompt:
            # 处理音频token
            assert value.shape[0] == self.audio_num_codebooks, "Number of codebooks mismatch"
            delta = HiggsAudioStreamerDelta(audio_tokens=value)
            # 直接使用主线程的循环，避免子线程动态获取
            self.loop.call_soon_threadsafe(self.queue.put_nowait, delta)
            return

        # 处理文本token
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        if len(value.shape) > 1:
            value = value[0]

        text = self.tokenizer.decode(value, **self.decode_kwargs)
        delta = HiggsAudioStreamerDelta(text=text, text_tokens=value)
        # 直接使用主线程的循环
        self.loop.call_soon_threadsafe(self.queue.put_nowait, delta)

    def end(self):
        """Flushes any remaining text tokens and signals the end of generation."""
        self.next_tokens_are_prompt = True
        # 直接使用主线程的循环
        self.loop.call_soon_threadsafe(self.queue.put_nowait, self.stop_signal)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            if self.has_asyncio_timeout:
                async with asyncio.timeout(self.timeout):
                    value = await self.queue.get()
            else:
                value = await asyncio.wait_for(self.queue.get(), timeout=self.timeout)
        except asyncio.TimeoutError:
            raise TimeoutError()
        else:
            if value == self.stop_signal:
                raise StopAsyncIteration()
            else:
                return value


class AsyncStoppingCriteria(StoppingCriteria):
    """
    Stopping criteria that checks for stop signal from a threading event.

    Args:
        stop_signal (threading.Event): Event that will receive stop signals
    """

    def __init__(self, stop_signal: threading.Event):
        self.stop_signal = stop_signal

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if self.stop_signal.is_set():
            logger.info(f"Stop signal received. Can be caused by client disconnection.")
            return True
        return False


class FastStopTokenCriteria(StoppingCriteria):
    def __init__(self, stop_token_ids, max_length):
        self.stop_token_ids = stop_token_ids  # 提前编码好的停止token ID列表
        self.max_length = max_length  # 最大生成长度（避免无限循环）

    def __call__(self, input_ids, scores, **kwargs):
        """
        适配你的_sample方法：接收input_ids（完整生成序列），判断是否停止
        """
        # 1. 达到最大长度，强制停止
        if input_ids.shape[1] >= self.max_length:
            return True
        # 2. 最后一个token是停止ID，停止（适配文本/音频生成的所有模式）
        last_token = input_ids[0, -1].item()  # CPU轻量操作，无同步
        return last_token in self.stop_token_ids


@dataclass
class HiggsAudioResponse:
    audio: Optional[np.ndarray] = None
    generated_audio_tokens: Optional[np.ndarray] = None
    sampling_rate: Optional[int] = None
    generated_text: str = ""
    generated_text_tokens: Optional[np.ndarray] = None
    usage: Optional[dict] = None


class HiggsAudioServeEngine:
    def __init__(
        self,
        model_name_or_path: str,
        audio_tokenizer_name_or_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        device: str = "cuda",
        torch_dtype: Union[torch.dtype, str] = "auto",
        kv_cache_lengths: List[int] = [1024, 4096, 8192],  # Multiple KV cache sizes
    ):
        """
        Initialize the HiggsAudioServeEngine, a serving wrapper for the HiggsAudioModel.
        The model, tokenizer, and audio tokenizer will be downloaded from the Hugging Face Hub if they are not local.

        Args:
            model_name_or_path (str):
                The name or path of the model to load.
            audio_tokenizer_name_or_path (str):
                The name or path of the audio tokenizer to load.
            tokenizer_name_or_path (str):
                The name or path of the tokenizer to load.
            device (str):
                The device to use for the model.
            kv_cache_lengths (List[int]):
                The lengths of the KV caches to use for the model. Used for cuda graph capture when device is cuda.
            torch_dtype (Union[torch.dtype, str]):
                The dtype to use for the model.
        """
        self.device = device
        self.model_name_or_path = model_name_or_path
        self.torch_dtype = torch_dtype

        # Initialize model and tokenizer
        # self.model = HiggsAudioModel.from_pretrained(model_name_or_path, torch_dtype=torch_dtype).to(device)


        cur_config = HiggsAudioConfig.from_pretrained(model_name_or_path)
        cur_config.text_config._attn_implementation = "sdpa"
        cur_config._attn_implementation = "sdpa"
        self.model = HiggsAudioModel.from_pretrained(
            config=cur_config,
            pretrained_model_name_or_path=model_name_or_path,
            torch_dtype=torch_dtype,
        ).to(device)
        logger.info(f"Loaded model from {model_name_or_path}, dtype: {self.model.dtype}")

        if tokenizer_name_or_path is None:
            tokenizer_name_or_path = model_name_or_path
        logger.info(f"Loading tokenizer from {tokenizer_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        logger.info(f"Initializing Higgs Audio Tokenizer")
        self.audio_tokenizer = load_higgs_audio_tokenizer(audio_tokenizer_name_or_path, device=device)

        self.audio_num_codebooks = self.model.config.audio_num_codebooks
        self.audio_codebook_size = self.model.config.audio_codebook_size
        self.audio_tokenizer_tps = self.audio_tokenizer.tps
        self.samples_per_token = int(self.audio_tokenizer.sampling_rate // self.audio_tokenizer_tps)
        self.hamming_window_len = 2 * self.audio_num_codebooks * self.samples_per_token
        # Set the audio special tokens
        self.model.set_audio_special_tokens(self.tokenizer)

        # Prepare KV caches for different lengths
        cache_config = deepcopy(self.model.config.text_config)
        cache_config.num_hidden_layers = self.model.config.text_config.num_hidden_layers
        if self.model.config.audio_dual_ffn_layers:
            cache_config.num_hidden_layers += len(self.model.config.audio_dual_ffn_layers)
        # A list of KV caches for different lengths
        self.kv_caches = {
            length: StaticCache(
                config=cache_config,
                max_batch_size=1,
                max_cache_len=length,
                device=self.model.device,
                dtype=self.model.dtype,
            )
            for length in sorted(kv_cache_lengths)
        }

        self.model.config.encode_whisper_embed = True
        self.model.config.skip_audio_tower = False

        if self.model.config.encode_whisper_embed:
            from transformers import WhisperProcessor
            whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        else:
            whisper_processor = None

        # Reuse collator to prepare inference samples
        self.collator = HiggsAudioSampleCollator(
            whisper_processor=whisper_processor,
            encode_whisper_embed=self.model.config.encode_whisper_embed,
            audio_in_token_id=self.model.config.audio_in_token_idx,
            audio_out_token_id=self.model.config.audio_out_token_idx,
            audio_stream_bos_id=self.model.config.audio_stream_bos_id,
            audio_stream_eos_id=self.model.config.audio_stream_eos_id,
            pad_token_id=self.model.config.pad_token_id,
            return_audio_in_tokens=True,  # must True!!!
            use_delay_pattern=self.model.config.use_delay_pattern,
            audio_num_codebooks=self.model.config.audio_num_codebooks,
            round_to=1,
        )

        # Capture CUDA graphs for each KV cache length
        # if device == "cuda":
        if "cuda" in device:
            logger.info(f"Capturing CUDA graphs for each KV cache length")
            self.model.capture_model(self.kv_caches.values())

    def _prepare_inputs(self, chat_ml_sample: ChatMLSample, force_audio_gen: bool = False):
        input_tokens, _, audio_contents,_, _ = prepare_chatml_sample(
            chat_ml_sample,
            self.tokenizer,
        )

        postfix = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        if force_audio_gen:
            postfix += "<|audio_out_bos|>"
        postfix = self.tokenizer.encode(postfix, add_special_tokens=False)
        input_tokens.extend(postfix)
        # print(input_tokens)
        # print(self.tokenizer.decode(input_tokens))
        # Configure the audio inputs
        audio_ids_l = []
        audio_waveforms_list, audio_sample_rates, audio_speaker_indices, audio_waveforms_start, current_waveform_offset = [], [], [], [], 0
        for audio_content in audio_contents:
            if audio_content.audio_url not in ["placeholder", ""]:
                # raw_audio, _ = librosa.load(audio_content.audio_url, sr=self.audio_tokenizer.sampling_rate)


                raw_audio, _ = torchaudio.load(audio_content.audio_url)
                if raw_audio.shape[0] > 1: raw_audio = raw_audio.mean(dim=0, keepdim=True)
                if _ != self.audio_tokenizer.sampling_rate: raw_audio = torchaudio.functional.resample(raw_audio, orig_freq=_, new_freq=self.audio_tokenizer.sampling_rate)
                raw_audio = raw_audio.squeeze(0)
            elif audio_content.raw_audio is not None:
                # raw_audio, _ = librosa.load(
                #     BytesIO(base64.b64decode(audio_content.raw_audio)), sr=self.audio_tokenizer.sampling_rate
                # )


                raw_audio, _ = torchaudio.load(BytesIO(base64.b64decode(audio_content.raw_audio)))
                if raw_audio.shape[0] > 1: raw_audio = raw_audio.mean(dim=0, keepdim=True)
                if _ != self.audio_tokenizer.sampling_rate: raw_audio = torchaudio.functional.resample(raw_audio, orig_freq=_, new_freq=self.audio_tokenizer.sampling_rate)
                raw_audio = raw_audio.squeeze(0)
            else:
                raw_audio = None

            if raw_audio is not None:
                audio_ids = self.audio_tokenizer.encode(raw_audio, self.audio_tokenizer.sampling_rate)
                audio_ids_l.append(audio_ids.squeeze(0).cpu())


                audio_waveforms_list.append(raw_audio)
                audio_length = len(raw_audio)
                audio_waveforms_start.append(current_waveform_offset)
                current_waveform_offset += audio_length

                audio_sample_rates.append(_)
                audio_speaker_indices.append(0)

        if len(audio_ids_l) > 0:
            audio_ids_start = torch.tensor(
                np.cumsum(np.array([0] + [audio_ids.shape[1] for audio_ids in audio_ids_l])),
                dtype=torch.long,
                device=self.device,
            )[0:-1]
            audio_ids_concat = torch.cat(audio_ids_l, dim=1)
        else:
            audio_ids_start = None
            audio_ids_concat = None

        sample = ChatMLDatasetSample(
            input_ids=torch.LongTensor(input_tokens),
            label_ids=None,
            audio_ids_concat=audio_ids_concat,
            audio_ids_start=audio_ids_start,
            # audio_waveforms_concat=None,
            # audio_waveforms_start=None,
            # audio_sample_rate=None,
            # audio_speaker_indices=None,
            audio_waveforms_concat=torch.cat(audio_waveforms_list, dim=0) if len(audio_waveforms_list) > 0 else None,
            audio_waveforms_start=torch.tensor(audio_waveforms_start, dtype=torch.long) if len(audio_waveforms_start) > 0 else None,
            audio_sample_rate=torch.tensor(audio_sample_rates, dtype=torch.float32) if len(audio_sample_rates) > 0 else None,
            audio_speaker_indices=torch.tensor(audio_speaker_indices, dtype=torch.long) if len(audio_speaker_indices) > 0 else None,
        )
        data = self.collator([sample])
        inputs = asdict(data)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.model.device)

        return inputs

    def _prepare_kv_caches(self):
        for kv_cache in self.kv_caches.values():
            kv_cache.reset()

    def generate_cold(
        self,
        chat_ml_sample: ChatMLSample,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: float = 0.95,
        stop_strings: Optional[List[str]] = None,
        force_audio_gen: bool = False,
        ras_win_len: Optional[int] = 7,
        ras_win_max_num_repeat: int = 2,
        seed: Optional[int] = None,
        streamer: Optional[AsyncHiggsAudioStreamer] = None,  # 新增streamer参数
    ):
        # Default stop strings
        if stop_strings is None:
            stop_strings = ["<|end_of_text|>", "<|eot_id|>"]


        cached_stop_token_ids = []
        for s in stop_strings:
            # 编码停止字符串，取完整token ID（你的stop_strings是单个标记，直接取全部）
            token_ids = self.tokenizer.encode(s, add_special_tokens=False)
            if token_ids:
                cached_stop_token_ids.extend(token_ids)  # 直接添加所有编码后的ID

        # 4. 创建自定义轻量停止条件（替代框架的StopStringCriteria）
        stopping_criteria = StoppingCriteriaList([
            FastStopTokenCriteria(
                stop_token_ids=cached_stop_token_ids,
                max_length=max_new_tokens
            )
        ])
        if ras_win_len is not None and ras_win_len <= 0:
            ras_win_len = None
        with torch.no_grad():
            inputs = self._prepare_inputs(chat_ml_sample, force_audio_gen=force_audio_gen)

        return

    def generate(
        self,
        chat_ml_sample: ChatMLSample,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: float = 0.95,
        stop_strings: Optional[List[str]] = None,
        force_audio_gen: bool = False,
        ras_win_len: Optional[int] = 7,
        ras_win_max_num_repeat: int = 2,
        seed: Optional[int] = None,
        streamer: Optional[AsyncHiggsAudioStreamer] = None,  # 新增streamer参数
    ):
        """
        Generate audio from a chatml sample.
        Args:
            chat_ml_sample: A chatml sample.
            max_new_tokens: The maximum number of new tokens to generate.
            temperature: The temperature to use for the generation.
            top_p: The top p to use for the generation.
            stop_strings: A list of strings to stop the generation.
            force_audio_gen: Whether to force audio generation. This ensures the model generates audio tokens rather than text tokens.
            ras_win_len: The length of the RAS window. We use 7 by default. You can disable it by setting it to None or <=0.
            ras_win_max_num_repeat: The maximum number of times to repeat the RAS window.
        Returns:
            A dictionary with the following keys:
                audio: The generated audio.
                sampling_rate: The sampling rate of the generated audio.
        """
        # Default stop strings
        if stop_strings is None:
            stop_strings = ["<|end_of_text|>", "<|eot_id|>"]


        cached_stop_token_ids = []
        for s in stop_strings:
            # 编码停止字符串，取完整token ID（你的stop_strings是单个标记，直接取全部）
            token_ids = self.tokenizer.encode(s, add_special_tokens=False)
            if token_ids:
                cached_stop_token_ids.extend(token_ids)  # 直接添加所有编码后的ID

        # 4. 创建自定义轻量停止条件（替代框架的StopStringCriteria）
        stopping_criteria = StoppingCriteriaList([
            FastStopTokenCriteria(
                stop_token_ids=cached_stop_token_ids,
                max_length=max_new_tokens
            )
        ])


        if ras_win_len is not None and ras_win_len <= 0:
            ras_win_len = None

        with torch.no_grad():
            time_before_prepare_inputs = time.time()
            inputs = self._prepare_inputs(chat_ml_sample, force_audio_gen=force_audio_gen)
            time_after_prepare_inputs = time.time()
            print(f'Time cost prepare_inputs: {time_after_prepare_inputs - time_before_prepare_inputs}')
            prompt_token_ids = inputs["input_ids"][0].cpu().numpy()

            self._prepare_kv_caches()

            time_after_prepare_kv_caches = time.time()
            print(f'Time cost prepare_kv_caches: {time_after_prepare_kv_caches - time_after_prepare_inputs}')

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stop_strings=stop_strings,
                tokenizer=self.tokenizer,
                do_sample=False if temperature == 0.0 else True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                past_key_values_buckets=self.kv_caches,
                ras_win_len=ras_win_len,
                ras_win_max_num_repeat=ras_win_max_num_repeat,
                seed=seed,
                streamer=streamer,  # 传入streamer
                stopping_criteria=stopping_criteria,  # 传入stopping_criteria
            )

            time_after_generate = time.time()
            print(f'Time cost generate: {time_after_generate - time_after_prepare_kv_caches}')

        # # 保留基础返回（streamer已实时输出，这里可简化）
        # generated_text_tokens = outputs[0][0].cpu().numpy()[len(prompt_token_ids) :]
        # generated_text = self.tokenizer.decode(generated_text_tokens)
        # generated_audio_tokens = outputs[1][0].cpu().numpy() if len(outputs[1]) > 0 else None
        
        # return HiggsAudioResponse(
        #     audio=None,  # 音频通过streamer实时处理，无需在这里返回
        #     generated_audio_tokens=generated_audio_tokens,
        #     sampling_rate=self.audio_tokenizer.sampling_rate,
        #     generated_text=generated_text,
        #     generated_text_tokens=generated_text_tokens,
        #     usage={
        #         "prompt_tokens": prompt_token_ids.shape[0],
        #         "completion_tokens": generated_text_tokens.shape[0] + (generated_audio_tokens.shape[1] if generated_audio_tokens is not None else 0),
        #         "total_tokens": prompt_token_ids.shape[0] + generated_text_tokens.shape[0] + (generated_audio_tokens.shape[1] if generated_audio_tokens is not None else 0),
        #         "cached_tokens": 0,
        #     },
        # )
