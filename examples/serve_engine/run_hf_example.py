"""Example for using HiggsAudio for generating both the transcript and audio in an interleaved manner."""

import os

from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
import torch
import torchaudio
import time
from loguru import logger
import click

from input_samples import INPUT_SAMPLES

MODEL_PATH = "./higgs-audio-v2-generation-3B-base"
# MODEL_PATH = "/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/train-higgs-audio/output_train/0828_multi_round_test_audio_label/checkpoint-2000"
# MODEL_PATH = "/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/train-higgs-audio/output_train/0830_multi_round/checkpoint-4500"
# MODEL_PATH = "/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/train-higgs-audio/output_train/0830_multi_round_only_audio/checkpoint-3888"
# MODEL_PATH = "/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/train-higgs-audio/output_train/0830_multi_round_voice_cloning/checkpoint-5182"


MODEL_DICT = {
    "0903": {
        "MODEL_ID": "0903_multi_round",
        "STEP": 6802,
    },
    "0905": {
        "MODEL_ID": "0905_multi_round",
        "STEP": 8880,
    },
    "0906": {
        "MODEL_ID": "0906_multi_round",
        "STEP": 10220,
    },
    "0910": {
        "MODEL_ID": "0910_multi_round_one_epoch",
        "STEP": 18000,  # 23853
    },
    "0910_multi_servers": {
        "MODEL_ID": "0910_multi_round_multi_servers",
        "STEP": 16000,  # 
    },
    "0914": {
        "MODEL_ID": "0914_from_understanding_H20",
        "STEP": 8000,  # 
    }
}

DATE = "0914"
MODEL_ID = MODEL_DICT[DATE]["MODEL_ID"]
STEP = MODEL_DICT[DATE]["STEP"]

MODEL_PATH = f"/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/train-higgs-audio/output_train/{MODEL_ID}/checkpoint-{STEP}"
AUDIO_TOKENIZER_PATH = "./higgs-audio-v2-tokenizer"


@click.command()
@click.argument("example", type=click.Choice(list(INPUT_SAMPLES.keys())))
def main(example: str):
    input_sample = INPUT_SAMPLES[example](response_type="ta")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    serve_engine = HiggsAudioServeEngine(
        MODEL_PATH,
        AUDIO_TOKENIZER_PATH,
        device=device,
    )

    logger.info("Starting generation...")
    start_time = time.time()
    output: HiggsAudioResponse = serve_engine.generate(
        chat_ml_sample=input_sample,
        max_new_tokens=1024,
        temperature=1.0,  # 1.0
        top_p=0.8,  # 0.8
        top_k=2,  # 2
        stop_strings=["<|end_of_text|>", "<|eot_id|>"],
    )
    elapsed_time = time.time() - start_time
    logger.info(f"Generation time: {elapsed_time:.2f} seconds")

    save_dir = os.path.join("generations", MODEL_ID, f"{STEP}_steps")

    if output.audio is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torchaudio.save(os.path.join(save_dir, f"output_{example}.wav"), torch.from_numpy(output.audio)[None, :], output.sampling_rate)
        with open(os.path.join(save_dir, f"output_{example}.txt"), 'w', encoding='utf-8') as f:
            f.write(output.generated_text.replace("<|audio_out_bos|><|AUDIO_OUT|><|audio_eos|><|eot_id|>", ""))
    logger.info(f"Generated text:\n{output.generated_text}")
    if output.audio is not None:
        logger.info(f"Saved audio to output_{example}.wav")


if __name__ == "__main__":
    # import debugpy
    # debugpy.listen(("0.0.0.0", 9501))
    # print("Waiting for debugger attach")
    # debugpy.wait_for_client()
    main()
