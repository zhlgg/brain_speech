# Evaluate with Seed-TTS testset

import argparse
import json
import os
import sys

sys.path.append(os.getcwd())

import multiprocessing as mp
from importlib.resources import files

import numpy as np
from utils_eval import (
    get_seed_tts_test,
    get_asr_test_set,
    run_asr_only_whisper,
    run_asr_wer_whisper,
    run_sim,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--eval_task", type=str, default="wer", choices=["sim", "wer", "asr"])
    parser.add_argument("-l", "--lang", type=str, default="en", choices=["zh", "en"])
    parser.add_argument("-g", "--gen_wav_dir", type=str, required=True)
    parser.add_argument("-n", "--gpu_nums", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--local", action="store_true", help="Use local custom checkpoint directory")
    return parser.parse_args()


def main():
    args = get_args()
    eval_task = args.eval_task
    lang = args.lang
    gen_wav_dir = args.gen_wav_dir
    # metalst = "/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/F5-TTS/tests/seedtts_testset/en/meta.lst"
    # metalst = "/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/F5-TTS/tests/basetts/test_en1_prompt.lst"
    # metalst = "/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/train-higgs-audio/generations/paper_model/273000_steps/llama-questions/a_as_input.jsonl"
    # metalst = "/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/train-higgs-audio/generations/paper_model/125000_steps/triviaqa/a_as_input.jsonl"
    # metalst = "/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/train-higgs-audio/generations/paper_model/125000_steps/web-questions/a_as_input.jsonl"

    metalst = "/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/LLaMA-Factory/qwen25_omni/a_as_input.jsonl"

    # lock
    gen_wav_dir = "/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/train-higgs-audio"

    # NOTE. paraformer-zh result will be slightly different according to the number of gpus, cuz batchsize is different
    #       zh 1.254 seems a result of 4 workers wer_seed_tts
    gpus = list(range(args.gpu_nums))
    # test_set = get_seed_tts_test(metalst, gen_wav_dir, gpus)
    test_set = get_asr_test_set(metalst, gen_wav_dir, gpus)

    local = args.local
    if local:  # use local custom checkpoint dir
        if lang == "zh":
            asr_ckpt_dir = "/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/F5-TTS/ckpts/paraformer-zh"  # paraformer-zh dir under funasr
        elif lang == "en":
            asr_ckpt_dir = "/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/F5-TTS/ckpts/hubert-large-ls960-ft"
    else:
        asr_ckpt_dir = ""  # auto download to cache dir
    wavlm_ckpt_dir = "/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/F5-TTS/ckpts/wavlm_large_finetune.pth"

    # --------------------------- WER ---------------------------
    wer_result_path = f"{gen_wav_dir}/{lang}_wer_results.jsonl"
    # lock
    asr_result_path = metalst.replace("a_as_input.jsonl", "a_as_input_s2s.jsonl")


    if eval_task == "asr":
        asr_results = []
        with mp.Pool(processes=len(gpus)) as pool:
            # 构造参数（保持与你的代码一致，asr_ckpt_dir会被run_asr_only_whisper忽略）
            args = [(rank, lang, sub_test_set, asr_ckpt_dir) for (rank, sub_test_set) in test_set]
            results = pool.map(run_asr_only_whisper, args)
            for r in results:
                asr_results.extend(r)

        # 保存ASR结果（每行一个JSON对象，与输入格式一致）
        with open(asr_result_path, "w", encoding="utf-8") as f:
            for line in asr_results:
                json_line = json.dumps(line, ensure_ascii=False)
                f.write(json_line + "\n")

        # 输出ASR任务统计信息（无WER相关）
        print(f"\nASR Evaluation Completed!")
        print(f"Total processed samples: {len(asr_results)}")
        print(f"Successfully processed: {len([x for x in asr_results if not x['response'].startswith('ASR_ERROR')])}")
        print(f"Failed samples: {len([x for x in asr_results if x['response'].startswith('ASR_ERROR')])}")
        print(f"Results have been saved to: {asr_result_path}")

    if eval_task == "wer":
        wer_results = []
        wers = []
        with mp.Pool(processes=len(gpus)) as pool:
            args = [(rank, lang, sub_test_set, asr_ckpt_dir) for (rank, sub_test_set) in test_set]
            results = pool.map(run_asr_wer_whisper, args)
            for r in results:
                wer_results.extend(r)

        with open(wer_result_path, "w") as f:
            for line in wer_results:
                wers.append(line["wer"])
                json_line = json.dumps(line, ensure_ascii=False)
                f.write(json_line + "\n")

        wer = round(np.mean(wers) * 100, 3)
        print(f"\nTotal {len(wers)} samples")
        print(f"WER      : {wer}%")
        print(f"Results have been saved to {wer_result_path}")
        with open(wer_result_path, "a") as f:
            f.write(f"Total {len(wers)},WER      :{wer}%")

    # --------------------------- SIM ---------------------------

    if eval_task == "sim":
        sims = []
        with mp.Pool(processes=len(gpus)) as pool:
            args = [(rank, sub_test_set, wavlm_ckpt_dir) for (rank, sub_test_set) in test_set]
            results = pool.map(run_sim, args)
            for r in results:
                sims.extend(r)

        sim = round(sum(sims) / len(sims), 3)
        print(f"\nTotal {len(sims)} samples")
        print(f"SIM      : {sim}")
        print(f"Results have been saved to {wer_result_path}")
        with open(wer_result_path, "a") as f:
            f.write(f"Total {len(sims)},SIM      :{sim}%")

if __name__ == "__main__":
    main()
