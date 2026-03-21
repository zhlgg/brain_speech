"""Example for using HiggsAudio for generating both the transcript and audio in an interleaved manner."""

import os
import json

from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
import torch
import torchaudio
import time
from loguru import logger
import click

# edit accelerate
from accelerate import Accelerator
accelerator = Accelerator()
from accelerate.utils import gather_object

from boson_multimodal.model.higgs_audio.modeling_higgs_audio import HiggsAudioEncoder, AudioAligner

from boson_multimodal.data_types import ChatMLSample, Message, AudioContent

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
        "STEP": 51000,
    },
}

DATE = "1028_attn_last_out_81795_tts_emilia"
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

    print(serve_engine.model, 'see model')

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

    batch_size = 8
    responses = []
    audio_save_paths = []
    result_save_dir = os.path.join("generations", MODEL_ID, f"{STEP}_steps", dataset_flag)
    result_json_save_path = os.path.join(result_save_dir, f"{request_type}_as_input.jsonl")
    audio_save_dir = os.path.join(result_save_dir, f"{request_type}_as_input")

    for batch_start in range(0, len(audio_ids), batch_size):
    # for batch_start in range(0, batch_size, batch_size):
        batch_end = min(batch_start + batch_size, len(audio_ids))
        batch_requests = requests[batch_start:batch_end]
        batch_audio_ids = audio_ids[batch_start:batch_end]
        batch_questions = questions[batch_start:batch_end]

        logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(audio_ids)-1)//batch_size + 1}")

        batch_samples = []

        for cur_request in batch_requests:
            cur_sample = get_chat_ml(request_type=request_type, request=cur_request, response_type="ta")
            batch_samples.append(cur_sample)

        with accelerator.split_between_processes(batch_samples) as batch_samples:
            accelerator.wait_for_everyone()
            results = [serve_engine.generate(
                chat_ml_sample=batch_samples[0],
                max_new_tokens=4096,
                temperature=1.0,  # 1.0 / 2.0   1.0   1.0
                top_p=0.5,  # 0.8   0.9   0.5
                top_k=1,  # 2  3   2   1
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
            )]
            outputs = gather_object(results)

        elapsed_time = time.time() - start_time
        logger.info(f"Generation time: {elapsed_time:.2f} seconds")

        # Save results
        if accelerator.is_main_process:
            for i, (output, cur_audio_id, cur_question) in enumerate(zip(outputs, batch_audio_ids, batch_questions)):
                audio_save_path = os.path.join(audio_save_dir, f"{cur_audio_id}.wav")
                if not os.path.exists(audio_save_dir):
                    os.makedirs(audio_save_dir)
                if output.audio is not None:
                    torchaudio.save(audio_save_path, torch.from_numpy(output.audio)[None, :], output.sampling_rate)
                with open(os.path.join(audio_save_dir, f"{cur_audio_id}_in.txt"), 'w', encoding='utf-8') as f:
                    f.write(cur_question)
                with open(os.path.join(audio_save_dir, f"{cur_audio_id}_out.txt"), 'w', encoding='utf-8') as f:
                    cur_response = output.generated_text.replace("<|eot_id|>", "")
                    cur_response = cur_response.replace("<|audio_out_bos|><|AUDIO_OUT|><|audio_out_eos|>", "")
                    cur_response = cur_response.replace("<|audio_out_last_bos|><|AUDIO_OUT|><|audio_out_eos|>", "")
                    responses.append(cur_response)
                    f.write(cur_response)
                logger.info(f"Generated text:\n{output.generated_text}")
                if output.audio is not None:
                    logger.info(f"Saved audio to {audio_save_path}")
                audio_save_paths.append(audio_save_path)

    with open(result_json_save_path, 'w') as f:
        for audio_id, response, correct_response, audio_save_path in zip(audio_ids, responses, correct_responses, audio_save_paths):
            cur_result_dict = {
                'audio_id': audio_id,
                'response': response, 
                'correct_response': correct_response,
                'path': audio_save_path,
            }
            f.write(json.dumps(cur_result_dict, ensure_ascii=True) + '\n')

if __name__ == "__main__":
    dataset_dir = "VoiceBench/processed/llama-questions"
    audio_path = "audio"
    json_path = "llama-questions.json"
    request_type = "a"
    dataset_flag = "llama-questions"
    main(dataset_dir, audio_path, json_path, dataset_flag, request_type)
