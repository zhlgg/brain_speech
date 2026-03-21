import math
import re
import os
import random
import string
from pathlib import Path
from torchaudio.transforms import Resample
import torch
import torch.nn.functional as F
import torchaudio
import librosa
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import Wav2Vec2Processor, HubertForCTC
from ecapa_tdnn import ECAPA_TDNN_SMALL
from modules import MelSpec
from utils import convert_char_to_pinyin

def number_to_words(n):
    units = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    if n == 0:
        return units[0]

    words = []

    # 处理百万
    if n >= 1000000:
        millions = n // 1000000
        words.append(number_to_words(millions) + " million")
        n %= 1000000

    # 处理千位（递归）
    if n >= 1000:
        thousands = n // 1000
        words.append(number_to_words(thousands) + " thousand")
        n %= 1000
        if 0 < n < 100:
            words.append("and")

    # 处理百位
    if n >= 100:
        hundreds = n // 100
        words.append(units[hundreds] + " hundred")
        n %= 100
        if n > 0:
            words.append("and")

    # 处理十位和个位
    if n >= 20:
        words.append(tens[n // 10])
        n %= 10
    elif 10 <= n < 20:
        words.append(teens[n - 10])
        n = 0

    if n > 0:
        words.append(units[n])

    return " ".join(words).replace(" and zero", "").replace("  ", " ")

def replace_mixed_numbers(text):
    # 分割字符串为数字和非数字部分
    parts = re.findall(r'\d+|\D+', text)
    converted = []
    for part in parts:
        if part.isdigit():
            converted.append(number_to_words(int(part)))
        else:
            # 保留非数字部分（如字母、符号）
            converted.append(part)
    # 合并并标准化空格
    return re.sub(r'\s+', ' ', ' '.join(converted)).strip()

def replace_special(text):
    if "$" in text:
        text = text.replace("$", "")
        text += "dollars" 
    if "supercomputer" in text:
        text = text.replace("supercomputer", "super computer")
    if "18th" or "19th" in text:
        text = text.replace("18th", "eighteenth").replace("19th", "nineteenth")
    
    return text

def section_to_chinese(section):
    """将不超过四位的数字转换为中文"""
    digits = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
    units = ['', '十', '百', '千']
    result = ''
    length = len(section)
    zero_flag = False

    for i, ch in enumerate(section):
        num = int(ch)
        pos = length - i - 1

        if num == 0:
            zero_flag = True
        else:
            if zero_flag:
                result += '零'
                zero_flag = False
            result += digits[num] + units[pos]

    return result.rstrip('零')

def number_to_chinese(n):
    if n == 0:
        return '零'

    str_n = str(n)
    str_n = str_n.zfill(((len(str_n)-1)//4 + 1)*4)  # 填充成4位倍数
    sections = [str_n[i:i+4] for i in range(0, len(str_n), 4)]
    section_units = ['', '万', '亿', '兆']

    result = ''
    for i, sec in enumerate(sections):
        part = section_to_chinese(sec)
        if part:
            result += part + section_units[len(sections) - i - 1]

    # 处理以“零十…”开头的情况（如“零一十”应为“一十”）
    result = re.sub(r'^零+', '', result)
    result = re.sub(r'^一十', '十', result)

    return result

def replace_mixed_zhnumbers(text):
    parts = re.findall(r'\d+|\D+', text)
    converted = []
    for part in parts:
        if part.isdigit():
            try:
                converted.append(number_to_chinese(int(part)))
            except:
                converted.append(part)
        else:
            converted.append(part)
    return ''.join(converted)

def get_seedtts_testset_metainfo(metalst):
    f = open(metalst)
    lines = f.readlines()
    f.close()
    metainfo = []
    for line in lines:
        if len(line.strip().split("|")) == 5:
            utt, prompt_text, prompt_wav, gt_text, gt_wav = line.strip().split("|")
        elif len(line.strip().split("|")) == 4:
            utt, prompt_text, prompt_wav, gt_text = line.strip().split("|")
            gt_wav = os.path.join(os.path.dirname(metalst), "wavs", utt + ".wav")
        if not os.path.isabs(prompt_wav):
            prompt_wav = os.path.join(os.path.dirname(metalst), prompt_wav)
        metainfo.append((utt, prompt_text, prompt_wav, gt_text, gt_wav))
    return metainfo


# librispeech test-clean metainfo: gen_utt, ref_txt, ref_wav, gen_txt, gen_wav
def get_librispeech_test_clean_metainfo(metalst, librispeech_test_clean_path):
    f = open(metalst)
    lines = f.readlines()
    f.close()
    metainfo = []
    for line in lines:
        ref_utt, ref_dur, ref_txt, gen_utt, gen_dur, gen_txt = line.strip().split("\t")

        # ref_txt = ref_txt[0] + ref_txt[1:].lower() + '.'  # if use librispeech test-clean (no-pc)
        ref_spk_id, ref_chaptr_id, _ = ref_utt.split("-")
        ref_wav = os.path.join(librispeech_test_clean_path, ref_spk_id, ref_chaptr_id, ref_utt + ".flac")

        # gen_txt = gen_txt[0] + gen_txt[1:].lower() + '.'  # if use librispeech test-clean (no-pc)
        gen_spk_id, gen_chaptr_id, _ = gen_utt.split("-")
        gen_wav = os.path.join(librispeech_test_clean_path, gen_spk_id, gen_chaptr_id, gen_utt + ".flac")

        metainfo.append((gen_utt, ref_txt, ref_wav, " " + gen_txt, gen_wav))

    return metainfo


# padded to max length mel batch
def padded_mel_batch(ref_mels):
    max_mel_length = torch.LongTensor([mel.shape[-1] for mel in ref_mels]).amax()
    padded_ref_mels = []
    for mel in ref_mels:
        padded_ref_mel = F.pad(mel, (0, max_mel_length - mel.shape[-1]), value=0)
        padded_ref_mels.append(padded_ref_mel)
    padded_ref_mels = torch.stack(padded_ref_mels)
    padded_ref_mels = padded_ref_mels.permute(0, 2, 1)
    return padded_ref_mels
# get prompts from metainfo containing: utt, prompt_text, prompt_wav, gt_text, gt_wav
def get_inference_prompt(
    metalst,
    metainfo,
    speed=1.0,
    tokenizer="pinyin",
    polyphone=True,
    target_sample_rate=24000,
    n_fft=1024,
    win_length=1024,
    n_mel_channels=100,
    hop_length=256,
    mel_spec_type="vocos",
    target_rms=0.1,
    use_truth_duration=False,
    infer_batch_size=1,
    num_buckets=200,
    min_secs=1,
    max_secs=60,
):
    prompts_all = []

    min_tokens = min_secs * target_sample_rate // hop_length
    max_tokens = max_secs * target_sample_rate // hop_length

    batch_accum = [0] * num_buckets
    utts, ref_rms_list, ref_mels, ref_mel_lens, total_mel_lens, final_text_list = (
        [[] for _ in range(num_buckets)] for _ in range(6)
    )

    mel_spectrogram = MelSpec(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )
    res = []
    for utt, prompt_text, prompt_wav, gt_text, gt_wav in tqdm(metainfo, desc="Processing prompts..."):
        # Audio
        ref_audio, ref_sr = torchaudio.load(prompt_wav)
        ref_rms = torch.sqrt(torch.mean(torch.square(ref_audio)))
        if ref_rms < target_rms:
            ref_audio = ref_audio * target_rms / ref_rms
        assert ref_audio.shape[-1] > 5000, f"Empty prompt wav: {prompt_wav}, or torchaudio backend issue."
        if ref_sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(ref_sr, target_sample_rate)
            ref_audio = resampler(ref_audio)

        # Text
        if len(prompt_text[-1].encode("utf-8")) == 1:
            prompt_text = prompt_text + " "
        text = [prompt_text + gt_text]
        if tokenizer == "pinyin":
            text_list = convert_char_to_pinyin(text, polyphone=polyphone)
        else:
            text_list = text

        # Duration, mel frame length
        ref_mel_len = ref_audio.shape[-1] // hop_length
        if use_truth_duration:
            gt_audio, gt_sr = torchaudio.load(gt_wav)
            if gt_sr != target_sample_rate:
                resampler = torchaudio.transforms.Resample(gt_sr, target_sample_rate)
                gt_audio = resampler(gt_audio)
            total_mel_len = ref_mel_len + int(gt_audio.shape[-1] / hop_length / speed)

            # # test vocoder resynthesis
            # ref_audio = gt_audio
        else:
            ref_text_len = len(prompt_text.encode("utf-8"))
            gen_text_len = len(gt_text.encode("utf-8"))
            total_mel_len = ref_mel_len + int(ref_mel_len / ref_text_len * gen_text_len / speed)

        # to mel spectrogram
        ref_mel = mel_spectrogram(ref_audio)
        ref_mel = ref_mel.squeeze(0)

        # deal with batch
        assert infer_batch_size > 0, "infer_batch_size should be greater than 0."
        if total_mel_len < min_tokens or total_mel_len > max_tokens:
            print(f"Audio {utt} has duration {total_mel_len*hop_length//target_sample_rate}s out of range [{min_secs}, {max_secs}].")
            continue
        assert (
            min_tokens <= total_mel_len <= max_tokens
        ), f"Audio {utt} has duration {total_mel_len*hop_length//target_sample_rate}s out of range [{min_secs}, {max_secs}]."
        bucket_i = math.floor((total_mel_len - min_tokens) / (max_tokens - min_tokens + 1) * num_buckets)

        utts[bucket_i].append(utt)
        ref_rms_list[bucket_i].append(ref_rms)
        ref_mels[bucket_i].append(ref_mel)
        ref_mel_lens[bucket_i].append(ref_mel_len)
        total_mel_lens[bucket_i].append(total_mel_len)
        final_text_list[bucket_i].extend(text_list)

        batch_accum[bucket_i] += total_mel_len
        res.append((utt,prompt_text,prompt_wav,gt_text,gt_wav))
        if batch_accum[bucket_i] >= infer_batch_size:
            # print(f"\n{len(ref_mels[bucket_i][0][0])}\n{ref_mel_lens[bucket_i]}\n{total_mel_lens[bucket_i]}")
            prompts_all.append(
                (
                    utts[bucket_i],
                    ref_rms_list[bucket_i],
                    padded_mel_batch(ref_mels[bucket_i]),
                    ref_mel_lens[bucket_i],
                    total_mel_lens[bucket_i],
                    final_text_list[bucket_i],
                )
            )
            batch_accum[bucket_i] = 0
            (
                utts[bucket_i],
                ref_rms_list[bucket_i],
                ref_mels[bucket_i],
                ref_mel_lens[bucket_i],
                total_mel_lens[bucket_i],
                final_text_list[bucket_i],
            ) = [], [], [], [], [], []

    # add residual
    for bucket_i, bucket_frames in enumerate(batch_accum):
        if bucket_frames > 0:
            prompts_all.append(
                (
                    utts[bucket_i],
                    ref_rms_list[bucket_i],
                    padded_mel_batch(ref_mels[bucket_i]),
                    ref_mel_lens[bucket_i],
                    total_mel_lens[bucket_i],
                    final_text_list[bucket_i],
                )
            )
    # not only leave easy work for last workers
    # random.seed(666)
    # random.shuffle(prompts_all)
    ##save json
    ##utt, prompt_text, prompt_wav, gt_text, gt_wav
    fout =  open(metalst.replace('3.5M_CN_answer/','3.5M_CN_answer/generate/'), "w")
    for id, prompt_text, prompt_wav, gt_text, gt_wav in res:
        fout.write(f"{id}|{prompt_text}|{prompt_wav}|{gt_text}|{gt_wav}\n")

    return prompts_all


# get wav_res_ref_text of seed-tts test metalst
# https://github.com/BytedanceSpeech/seed-tts-eval
def get_seed_tts_test(metalst, gen_wav_dir, gpus):
    f = open(metalst)
    lines = f.readlines()
    f.close()

    test_set_ = []
    for line in tqdm(lines):
        if len(line.strip().split("|")) == 5:
            utt, prompt_text, prompt_wav, gt_text, gt_wav = line.strip().split("|")
        elif len(line.strip().split("|")) == 4:
            utt, prompt_text, prompt_wav, gt_text = line.strip().split("|")

        if not os.path.exists(os.path.join(gen_wav_dir, utt + ".wav")):
            continue
        gen_wav = os.path.join(gen_wav_dir, utt + ".wav")
        if not os.path.isabs(prompt_wav):
            prompt_wav = os.path.join(os.path.dirname(metalst), prompt_wav)

        test_set_.append((gen_wav, prompt_wav, gt_text))

    num_jobs = len(gpus)
    if num_jobs == 1:
        return [(gpus[0], test_set_)]

    wav_per_job = len(test_set_) // num_jobs + 1
    test_set = []
    for i in range(num_jobs):
        test_set.append((gpus[i], test_set_[i * wav_per_job : (i + 1) * wav_per_job]))

    return test_set

import json
def get_asr_test_set(json_file, gen_wav_dir, gpus):
    """
    生成ASR任务的test_set（处理JSON格式输入文件）
    Args:
        json_file: JSON输入文件路径（每行一个JSON对象）
        gpus: GPU ID列表（如[0,1,2]）
    Returns:
        按GPU拆分的test_set: [(gpu_id, [(audio_id, audio_path, correct_response), ...]), ...]
    """
    # 检查输入文件是否存在
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"输入文件不存在：{json_file}")
    
    # 读取JSON文件并解析
    test_set_ = []
    json_dir = os.path.dirname(json_file)  # JSON文件所在目录（用于处理相对路径）
    
    with open(json_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="解析输入文件"):
            line = line.strip()
            if not line:
                continue
            
            # 解析JSON对象
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"警告：跳过无效JSON行：{line}，错误：{e}")
                continue
            
            # 提取必要字段
            required_fields = ["audio_id", "correct_response", "path"]
            if not all(field in data for field in required_fields):
                print(f"警告：跳过缺少字段的JSON行：{line}")
                continue
            
            audio_id = data["audio_id"]
            correct_response = data["correct_response"]
            audio_path = data["path"]
            audio_path = os.path.join(gen_wav_dir, audio_path)
            
            # 检查音频文件是否存在
            if not os.path.exists(audio_path):
                print(f"警告：音频文件不存在，跳过：{audio_path}")
                continue
            
            # 添加到test_set（保存audio_id用于后续匹配，correct_response用于最终输出）
            test_set_.append((audio_id, audio_path, correct_response))
    
    # 按GPU数量拆分任务（保持与原始函数一致的多GPU处理逻辑）
    num_jobs = len(gpus)
    if num_jobs == 1:
        return [(gpus[0], test_set_)]
    
    # 平均分配任务（不足1个GPU的部分分配给最后一个GPU）
    wav_per_job = len(test_set_) // num_jobs
    remainder = len(test_set_) % num_jobs
    test_set = []
    
    start_idx = 0
    for i in range(num_jobs):
        end_idx = start_idx + wav_per_job + (1 if i < remainder else 0)
        test_set.append((gpus[i], test_set_[start_idx:end_idx]))
        start_idx = end_idx
    
    return test_set


# get librispeech test-clean cross sentence test
def get_librispeech_test(metalst, gen_wav_dir, gpus, librispeech_test_clean_path, eval_ground_truth=False):
    f = open(metalst)
    lines = f.readlines()
    f.close()

    test_set_ = []
    for line in tqdm(lines):
        ref_utt, ref_dur, ref_txt, gen_utt, gen_dur, gen_txt = line.strip().split("\t")

        if eval_ground_truth:
            gen_spk_id, gen_chaptr_id, _ = gen_utt.split("-")
            gen_wav = os.path.join(librispeech_test_clean_path, gen_spk_id, gen_chaptr_id, gen_utt + ".flac")
        else:
            if not os.path.exists(os.path.join(gen_wav_dir, gen_utt + ".wav")):
                raise FileNotFoundError(f"Generated wav not found: {gen_utt}")
            gen_wav = os.path.join(gen_wav_dir, gen_utt + ".wav")

        ref_spk_id, ref_chaptr_id, _ = ref_utt.split("-")
        ref_wav = os.path.join(librispeech_test_clean_path, ref_spk_id, ref_chaptr_id, ref_utt + ".flac")

        test_set_.append((gen_wav, ref_wav, gen_txt))

    num_jobs = len(gpus)
    if num_jobs == 1:
        return [(gpus[0], test_set_)]

    wav_per_job = len(test_set_) // num_jobs + 1
    test_set = []
    for i in range(num_jobs):
        test_set.append((gpus[i], test_set_[i * wav_per_job : (i + 1) * wav_per_job]))

    return test_set


def load_asr_model(lang, ckpt_dir=""):
    if lang == "zh":
        from funasr import AutoModel

        model = AutoModel(
            model=os.path.join(ckpt_dir, "paraformer-zh"),
            # vad_model = os.path.join(ckpt_dir, "fsmn-vad"),
            # punc_model = os.path.join(ckpt_dir, "ct-punc"),
            # spk_model = os.path.join(ckpt_dir, "cam++"),
            disable_update=True,
        )  # following seed-tts setting
    elif lang == "en":
        from faster_whisper import WhisperModel

        model_size = "large-v3" if ckpt_dir == "" else ckpt_dir
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
    return model


# WER Evaluation, the way Seed-TTS does


def run_asr_wer(args):
    rank, lang, test_set, ckpt_dir = args

    if lang == "zh":
        import zhconv

        torch.cuda.set_device(rank)
    elif lang == "en":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    else:
        raise NotImplementedError(
            "lang support only 'zh' (funasr paraformer-zh), 'en' (faster-whisper-large-v3), for now."
        )

    asr_model = load_asr_model(lang, ckpt_dir=ckpt_dir)


    punctuation_all = string.punctuation
    wer_results = []

    from jiwer import compute_measures

    for gen_wav, prompt_wav, truth in tqdm(test_set):
        if lang == "zh":
            res = asr_model.generate(input=gen_wav, batch_size_s=300, disable_pbar=True)
            hypo = res[0]["text"]
            hypo = zhconv.convert(hypo, "zh-cn")
        elif lang == "en":
            segments, _ = asr_model.transcribe(gen_wav, beam_size=5, language="en")
            hypo = ""
            for segment in segments:
                hypo = hypo + " " + segment.text

        raw_truth = truth
        raw_hypo = hypo

        for x in punctuation_all:
            truth = truth.replace(x, "")
            hypo = hypo.replace(x, "")

        truth = truth.replace("  ", " ")
        hypo = hypo.replace("  ", " ")

        if lang == "zh":
            truth = " ".join([x for x in truth])
            hypo = " ".join([x for x in hypo])
        elif lang == "en":
            truth = truth.lower()
            hypo = hypo.lower()

        measures = compute_measures(truth, hypo)
        wer = measures["wer"]

        # ref_list = truth.split(" ")
        # subs = measures["substitutions"] / len(ref_list)
        # dele = measures["deletions"] / len(ref_list)
        # inse = measures["insertions"] / len(ref_list)

        wer_results.append(
            {
                "wav": Path(gen_wav).stem,
                "truth": raw_truth,
                "hypo": raw_hypo,
                "wer": wer,
            }
        )

    return wer_results

# WER Evaluation, the way Seed-TTS does
def run_asr_wer_hubert(args):
    rank, lang, test_set, ckpt_dir = args
    if lang == "zh":
        import zhconv

        torch.cuda.set_device(rank)
    elif lang == "en":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    else:
        raise NotImplementedError(
            "lang support only 'zh' (funasr paraformer-zh), 'en' (faster-whisper-large-v3), for now."
        )
    ckpt_dir = "/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/F5-TTS/ckpts/hubert-large-ls960-ft"
    asr_processor = Wav2Vec2Processor.from_pretrained(ckpt_dir)
    asr_model = HubertForCTC.from_pretrained(ckpt_dir).to("cuda").eval()
    punctuation_all =  string.punctuation
    wer_results = []

    from jiwer import compute_measures,cer
    # test_set = test_set[0]
    # rank, test_set = test_set[0], test_set[1]
    for gen_wav, prompt_wav, truth in tqdm(test_set):
        if lang == "zh":
            res = asr_model.generate(input=gen_wav, batch_size_s=300, disable_pbar=True)
            hypo = res[0]["text"]
            hypo = zhconv.convert(hypo, "zh-cn")
        elif lang == "en":
            tgt_sr = 16000
            wav, sr = torchaudio.load(gen_wav)   
            wav = wav.to("cuda")
            if sr != tgt_sr:
                resampler = Resample(sr, tgt_sr).to('cuda')
                wav = resampler(wav)
            input_values = asr_processor(wav[0], return_tensors="pt", sampling_rate=tgt_sr).input_values.to('cuda')  # Batch size 1
            logits = asr_model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            hypo = asr_processor.decode(predicted_ids[0])

        raw_truth = truth
        raw_hypo = hypo

        for x in punctuation_all:
            truth = truth.replace(x, "")
            hypo = hypo.replace(x, "")
        truth = truth.replace("  ", " ")
        hypo = hypo.replace("  ", " ")
        hypo = re.sub(r'[^\w\s\']', '', hypo)
        if lang == "zh":
            truth = " ".join([x for x in truth])
            hypo = " ".join([x for x in hypo])
        elif lang == "en":
            truth = truth.lower()
            hypo = hypo.lower()
        hypo = replace_mixed_numbers(hypo)
        hypo = replace_special(hypo)
        measures = compute_measures(truth, hypo)
        wer = measures["wer"]
        cer_pred = cer(truth, hypo)

        # ref_list = truth.split(" ")
        # subs = measures["substitutions"] / len(ref_list)
        # dele = measures["deletions"] / len(ref_list)
        # inse = measures["insertions"] / len(ref_list)

        wer_results.append(
            {
                "wav": Path(gen_wav).stem,
                "truth": raw_truth,
                "hypo": raw_hypo,
                "wer": wer,
                "cer": cer_pred,
            }
        )

    return wer_results

# WER Evaluation, the way Seed-TTS does
def run_asr_wer_whisper(args, is_ellav=True):
    rank, lang, test_set, ckpt_dir = args

    if lang == "zh":
        import zhconv

        torch.cuda.set_device(rank)
    elif lang == "en":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    else:
        raise NotImplementedError(
            "lang support only 'zh' (funasr paraformer-zh), 'en' (faster-whisper-large-v3), for now."
        )

    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to("cuda")
    wer_results = []

    from jiwer import wer,cer
    # test_set = test_set[0]
    # rank, test_set = test_set[0], test_set[1]

    for gen_wav, prompt_wav, truth in tqdm(test_set):
        if lang == "zh":
            res = asr_model.generate(input=gen_wav, batch_size_s=300, disable_pbar=True)
            hypo = res[0]["text"]
            hypo = zhconv.convert(hypo, "zh-cn")
        elif lang == "en":
            wav, sr = librosa.load(gen_wav, sr=16000)
            # hypo = asr_model.transcribe(
            #     wav,
            #     language="en",
            #     fp16=True
            # )
            input_features = processor(
                wav, sampling_rate=16000, return_tensors="pt"
            ).input_features
            input_features = input_features.to("cuda")
            forced_decoder_ids = processor.get_decoder_prompt_ids(
                language="english", task="transcribe"
            )
            predicted_ids = model.generate(
                input_features, forced_decoder_ids=forced_decoder_ids
            )
            hypo = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]

        raw_truth = truth
        raw_hypo = hypo

        truth = truth.replace('-', ' ')
        hypo = hypo.replace('-', ' ')
        truth = truth.replace('“', ' ').replace('”', ' ')
        hypo = hypo.replace('“', ' ').replace('”', ' ')
        import unicodedata, re

        def strip_punctuation(text: str) -> str:
            out = []
            for ch in text:
                cat = unicodedata.category(ch)
                # P* = punctuation, S* = symbol
                out.append(' ' if cat.startswith(('P', 'S')) else ch)
            return re.sub(r'\s+', ' ', ''.join(out)).strip()

        truth = strip_punctuation(truth)
        hypo  = strip_punctuation(hypo)

        if lang == "zh":
            truth = " ".join([x for x in truth])
            hypo = " ".join([x for x in hypo])
        elif lang == "en":
            truth = truth.lower()
            hypo = hypo.lower()
        # if is_ellav:
        hypo = replace_mixed_numbers(hypo)
        wer_pred = wer(truth, hypo,)
        cer_pred = cer(truth, hypo)
        
        wer_results.append(
            {
                "wav": Path(gen_wav).stem,
                "truth": raw_truth,
                "hypo": raw_hypo,
                "wer": wer_pred,
                "cer": cer_pred,
            }
        )
    return wer_results

def run_asr_only_whisper(args, is_ellav=True):
    """
    仅执行ASR推理（不计算WER/CER），返回更新后的JSON数据
    Args:
        args: 元组 (rank, lang, test_set, ckpt_dir)
            rank: GPU ID
            lang: 语言类型（"zh"或"en"）
            test_set: 该GPU需要处理的任务列表 [(audio_id, audio_path, correct_response), ...]
            ckpt_dir: 模型 checkpoint 目录（Whisper从Hugging Face加载，该参数仅为兼容原始接口）
        is_ellav: 是否使用ellav相关处理（保留原始参数，无实际作用）
    Returns:
        更新后的JSON数据列表：[{"audio_id": ..., "response": ASR结果, "correct_response": ..., "path": ...}, ...]
    """
    rank, lang, test_set, _ = args  # 忽略ckpt_dir，Whisper从Hugging Face加载
    
    # 设置GPU
    if lang == "zh":
        import zhconv
        torch.cuda.set_device(rank)
    elif lang == "en":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    else:
        raise NotImplementedError(
            "lang support only 'zh' (with zhconv), 'en' (faster-whisper-large-v3), for now."
        )
    
    # 加载Whisper模型和处理器（与原始代码一致）
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to("cuda")
    asr_results = []
    
    # 处理每个音频文件
    for audio_id, audio_path, correct_response in tqdm(test_set, desc=f"GPU {rank} 执行ASR"):
        try:
            if lang == "zh":
                # 原始代码中中文使用funasr，这里保持兼容（若你使用Whisper处理中文，可替换为英文逻辑）
                # 注意：需要提前安装funasr并初始化asr_model（示例如下，若无需可替换为Whisper逻辑）
                # ========== 方案1：使用funasr（与原始代码一致） ==========
                # from funasr import AutoModel
                # asr_model = AutoModel(model="paraformer-zh", model_revision="v2.0.4", device=f"cuda:{rank}")
                # res = asr_model.generate(input=audio_path, batch_size_s=300, disable_pbar=True)
                # hypo = res[0]["text"]
                # hypo = zhconv.convert(hypo, "zh-cn")  # 转为简体中文
                
                # ========== 方案2：使用Whisper处理中文（推荐，统一模型） ==========
                wav, sr = librosa.load(audio_path, sr=16000)
                input_features = processor(wav, sampling_rate=16000, return_tensors="pt").input_features.to("cuda")
                forced_decoder_ids = processor.get_decoder_prompt_ids(language="chinese", task="transcribe")
                predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
                hypo = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                hypo = zhconv.convert(hypo, "zh-cn")  # 转为简体中文
                
            elif lang == "en":
                # 英文处理逻辑（与原始代码一致）
                wav, sr = librosa.load(audio_path, sr=16000)
                input_features = processor(wav, sampling_rate=16000, return_tensors="pt").input_features.to("cuda")
                forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
                predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
                hypo = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            # 应用数字格式替换（如果需要）
            if is_ellav:
                hypo = replace_mixed_numbers(hypo)
            
            # 保存更新后的结果（保持原始JSON格式）
            asr_results.append({
                "audio_id": audio_id,
                "response": hypo.strip(),  # 用ASR结果替换原response字段
                "correct_response": correct_response,  # 保留原始correct_response
                "path": audio_path  # 保留原始音频路径（可改为相对路径，若需要）
            })
        
        except Exception as e:
            print(f"警告：处理音频 {audio_path} 时出错：{e}，跳过该文件")
            # 出错时保留原始数据（response标注为错误）
            asr_results.append({
                "audio_id": audio_id,
                "response": f"ASR_ERROR: {str(e)}",
                "correct_response": correct_response,
                "path": audio_path
            })
    
    return asr_results


def run_sim(args):
    rank, test_set, ckpt_dir = args
    device = f"cuda:{rank}"

    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    state_dict = torch.load(ckpt_dir, weights_only=True, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict["model"], strict=False)

    use_gpu = True if torch.cuda.is_available() else False
    if use_gpu:
        model = model.cuda(device)
    model.eval()

    sims = []
    for wav1, wav2, truth in tqdm(test_set):
        wav1, sr1 = torchaudio.load(wav1)
        wav2, sr2 = torchaudio.load(wav2)

        resample1 = torchaudio.transforms.Resample(orig_freq=sr1, new_freq=16000)
        resample2 = torchaudio.transforms.Resample(orig_freq=sr2, new_freq=16000)
        wav1 = resample1(wav1)
        wav2 = resample2(wav2)

        if use_gpu:
            wav1 = wav1.cuda(device)
            wav2 = wav2.cuda(device)
        with torch.no_grad():
            emb1 = model(wav1)
            emb2 = model(wav2)

        sim = F.cosine_similarity(emb1, emb2)[0].item()
        # print(f"VSim score between two audios: {sim:.4f} (-1.0, 1.0).")
        sims.append(sim)

    return sims
