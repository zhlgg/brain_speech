export PYTHONPATH=src:$PYTHONPATH

# python3 examples/generation.py \
#     --transcript "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years." \
#     --ref_audio broom_salesman \
#     --ref_audio_in_system_message \
#     --temperature 0.3 \
#     --out_path generation.wav

# python examples/serve_engine/run_hf_example.py interleaved_dialogue
python examples/serve_engine/run_hf_example.py chat
# python examples/serve_engine/run_test_multi_GPUs.py

# python3 examples/generation.py \
# --transcript examples/transcript/multi_speaker/en_argument.txt \
# --ref_audio belinda,broom_salesman \
# --ref_audio_in_system_message \
# --chunk_method speaker \
# --seed 12345 \
# --out_path generation.wav

# python3 examples/generation.py \
# --transcript examples/transcript/multi_speaker/en_higgs.txt \
# --ref_audio broom_salesman,belinda \
# --ref_audio_in_system_message \
# --chunk_method speaker \
# --chunk_max_num_turns 2 \
# --seed 12345 \
# --out_path generation.wav

# python3 examples/generation_answer.py \
# --transcript "Certainly. Let me see. Oh, it's on that shelf." \
# --model_path "output_train/0827_1/models" \
# --question_audio "/apdcephfs_cq10/share_1297902/data/speech_data/qa_data/dailytalk/data/11/2_1_d11.wav" \
# --seed 12345 \
# --out_path answer.wav

# taiji_client exec 8V100_f5_ppg_mt 8b1d81e196ec885001970fd427d251fe bash
# taiji_client exec 8V100_f5_txt 8b1d804b96a46cd60196a8c10fe50e69 bash
# taiji_client exec 16A100_f5_durpred 8b1d804b96a46cd60196a8c438cf0e79 bash
# taiji_client exec train_16V100_MT 8b1d81239710897f01971ab673f01811 bash
# taiji_client exec train_24V100_MT_2 8b1d81239710897f01971ab2a6ae1803 bash

# taiji_client exec 8H20_F5_omni_T 8b1d803a99248b7c0199326ab0ca1594 bash
# taiji_client exec 8H20_F5_omni_T1 8b1d807597447b020197732920af519a bash
# taiji_client exec 8H20_F5_omni_T2 8b1d821597108d79019734afa5f03aad bash
nohup bash from_understanding_cu124.sh > libritts_train.log 2>&1 &
# cd /apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/train-higgs-audio
# conda activate higgs


# taiji_client exec train_8A100_new_group1 8b1d810197447b12019752e6f59e0f48 bash
# taiji_client exec train_8A100_new_group3 8b1d807597447b020197522cdc7f0d52 bash
# taiji_client exec train_24V100_MT_2 8b1d81239710897f01971ab2a6ae1803 bash
# taiji_client exec 8H20_F5_omni_T2 8b1d821597108d79019734afa5f03aad bash
# taiji_client exec 16train_MT_V100 8b1d800a97108e66019734f266ec3b98 bash
# taiji_client exec 16V100_f5_cb_align 8b1d804b96a46cd60196a8c1f8790e6a bash
# /apdcephfs_cq10/share_1297902/data/speech_data/qa_data/minmax_instructs2s/higgs/answer_242859.wav
