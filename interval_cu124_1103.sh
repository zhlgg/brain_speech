export http_proxy=http://star-proxy.oa.com:3128 
export https_proxy=http://star-proxy.oa.com:3128
export http_proxy="http://9.21.0.122:11113"
export https_proxy="http://9.21.0.122:11113"

export NCCL_P2P_LEVEL=NVL

set -x
export CUDA_HOME=/usr/local/cuda-12.4/
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:LD_LIBRARY_PATH$


NET_TYPE="high"
export OMP_NUM_THREADS=8
export NCCL_IB_TIMEOUT=24


export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
if [[ "${NET_TYPE}" = "low" ]]; then
    export NCCL_SOCKET_IFNAME=eth1
    export NCCL_IB_DISABLE=1
    export NCCL_IB_GID_INDEX=3
    nccl_ib_hca=$(bash show_gids |  grep $(hostname -I) |  grep v2 | awk '{print $1 ":" $2}' )

    export NCCL_IB_HCA=mlx5_2:1,mlx5_2:1
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=1
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
else
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
    export NCCL_SOCKET_IFNAME=bond1
    export UCX_NET_DEVICES=bond1
    export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
    export NCCL_COLLNET_ENABLE=0
    export SHARP_COLL_ENABLE_SAT=0
    export NCCL_NET_GDR_LEVEL=2
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_IB_TC=160
    export NCCL_PXN_DISABLE=1
fi
node_num=${HOST_NUM}
n_gpus_per_node=${HOST_GPU_NUM}
node_rank=${INDEX:-0}
n_gpu=${HOST_GPU_NUM:-1}
port=2352
n_gpus=$((node_num * n_gpus_per_node))

cat /etc/hosts
echo "ngpus: ${n_gpus}"
echo "Hosts: ${HOST_NUM}"
echo "node_rank: ${node_rank}"
echo "master: ${CHIEF_IP}:${port}"
echo "local ip: ${LOCAL_IP}"   
echo "start infer..."


export PYTHONPATH=src:$PYTHONPATH
export NCCL_LAUNCH_MODE=GROUP


# # here understanding!!!
# torchrun \
#   ${HOST_NUM:+--nnodes ${HOST_NUM}} \
#   --nproc_per_node ${n_gpus_per_node} \
#   --node-rank ${node_rank} \
#   ${CHIEF_IP:+--master-addr ${CHIEF_IP}} \
#   --master_port=${port} \
#   --local-addr ${LOCAL_IP} \
#   trainer/trainer_ddp_multi_round.py \
#   --model_path higgs-audio-v2-generation-3B-base \
#   --audio_tokenizer_path higgs-audio-v2-tokenizer \
#   --train_data_dir higgs_training_data_mini/ \
#   --task_type multi_speaker_chat \
#   --output_dir output_train/1028_attn_last_out_understanding \
#   --per_device_train_batch_size 2 \
#   --num_train_epochs 3 \
#   --save_steps 1000 \
#   --bf16 \
#   --model_init \
#   --freeze_audio_tower \
#   --ta_rate 0.8


# torchrun \
#   ${HOST_NUM:+--nnodes ${HOST_NUM}} \
#   --nproc_per_node ${n_gpus_per_node} \
#   --node-rank ${node_rank} \
#   ${CHIEF_IP:+--master-addr ${CHIEF_IP}} \
#   --master_port=${port} \
#   --local-addr ${LOCAL_IP} \
#   trainer/trainer_ddp_multi_round.py \
#   --model_path higgs-audio-v2-generation-3B-base \
#   --audio_tokenizer_path higgs-audio-v2-tokenizer \
#   --train_data_dir higgs_training_data_mini/ \
#   --task_type multi_speaker_chat \
#   --step 1 \
#   --output_dir output_train/1117_ds_1_30_30_10_10_40_40_understanding_edit_ds \
#   --per_device_train_batch_size 2 \
#   --num_train_epochs 3 \
#   --warmup_steps 50 \
#   --save_steps 1000 \
#   --bf16 \
#   --model_init \
#   --freeze_audio_tower \
#   --deepspeed_file deepspeed_files/ds_config_zero2.json \
#   --ta_rate 0.8



# # here tts!!!
# torchrun --nproc_per_node=8 --master_port 29029 trainer/trainer_ddp_multi_round.py \
# torchrun \
#   ${HOST_NUM:+--nnodes ${HOST_NUM}} \
#   --nproc_per_node ${n_gpus_per_node} \
#   --node-rank ${node_rank} \
#   ${CHIEF_IP:+--master-addr ${CHIEF_IP}} \
#   --master_port=${port} \
#   --local-addr ${LOCAL_IP} \
#   trainer/trainer_ddp_multi_round.py \
#   --model_path output_train/1028_attn_last_out_understanding/checkpoint-81795 \
#   --audio_tokenizer_path higgs-audio-v2-tokenizer \
#   --train_data_dir higgs_training_data_mini/ \
#   --task_type multi_speaker_chat \
#   --output_dir output_train/1028_attn_last_out_81795_tts_emilia \
#   --per_device_train_batch_size 2 \
#   --num_train_epochs 3 \
#   --save_steps 1000 \
#   --bf16 \
#   --learning_rate 3e-4 \
#   --freeze_llm \
#   --freeze_audio_tower \
#   --freeze_audio_encoder_proj \
#   --freeze_embed \
#   --freeze_text_added_tokens \
#   --ta_rate 0.8


# torchrun --nproc_per_node=8 --master_port 29029 trainer/trainer_ddp_multi_round.py \
# torchrun \
#   ${HOST_NUM:+--nnodes ${HOST_NUM}} \
#   --nproc_per_node ${n_gpus_per_node} \
#   --node-rank ${node_rank} \
#   ${CHIEF_IP:+--master-addr ${CHIEF_IP}} \
#   --master_port=${port} \
#   --local-addr ${LOCAL_IP} \
#   trainer/trainer_ddp_multi_round.py \
#   --model_path output_train/1115_09_30_30_10_10_40_40_understanding/checkpoint-40899 \
#   --audio_tokenizer_path higgs-audio-v2-tokenizer \
#   --train_data_dir higgs_training_data_mini/ \
#   --task_type multi_speaker_chat \
#   --step 2 \
#   --output_dir output_train/1115_09_30_30_10_10_40_40_tts \
#   --per_device_train_batch_size 2 \
#   --num_train_epochs 3 \
#   --warmup_steps 250 \
#   --save_steps 1000 \
#   --bf16 \
#   --learning_rate 3e-4 \
#   --freeze_llm \
#   --freeze_audio_tower \
#   --freeze_audio_encoder_proj \
#   --freeze_embed \
#   --freeze_text_added_tokens \
#   --deepspeed_file deepspeed_files/ds_config_zero2.json \
#   --ta_rate 0.8


# # continue pre-train
# torchrun \
#   ${HOST_NUM:+--nnodes ${HOST_NUM}} \
#   --nproc_per_node ${n_gpus_per_node} \
#   --node-rank ${node_rank} \
#   ${CHIEF_IP:+--master-addr ${CHIEF_IP}} \
#   --master_port=${port} \
#   --local-addr ${LOCAL_IP} \
#   trainer/trainer_ddp_multi_round.py \
#   --model_path output_train/1115_1_30_30_10_10_40_40_tts_emilia_no_ds/checkpoint-322000 \
#   --audio_tokenizer_path higgs-audio-v2-tokenizer \
#   --train_data_dir higgs_training_data_mini/ \
#   --task_type multi_speaker_chat \
#   --step 2 \
#   --output_dir output_train/1204_continue_pretrain_from_32w \
#   --per_device_train_batch_size 2 \
#   --num_train_epochs 3 \
#   --warmup_steps 500 \
#   --save_steps 1000 \
#   --continue_pretrain \
#   --bf16 \
#   --learning_rate 3e-4 \
#   --freeze_llm \
#   --freeze_audio_tower \
#   --freeze_audio_encoder_proj \
#   --freeze_embed \
#   --freeze_text_added_tokens \
#   --deepspeed_file deepspeed_files/ds_config_zero2.json \
#   --ta_rate 0.8


# torchrun \
#   ${HOST_NUM:+--nnodes ${HOST_NUM}} \
#   --nproc_per_node ${n_gpus_per_node} \
#   --node-rank ${node_rank} \
#   ${CHIEF_IP:+--master-addr ${CHIEF_IP}} \
#   --master_port=${port} \
#   --local-addr ${LOCAL_IP} \
#   trainer/trainer_ddp_multi_round.py \
#   --model_path output_train/paper_model/checkpoint-320000 \
#   --audio_tokenizer_path higgs-audio-v2-tokenizer \
#   --train_data_dir higgs_training_data_mini/ \
#   --task_type multi_speaker_chat \
#   --step 3 \
#   --output_dir output_train/step_3_1208 \
#   --per_device_train_batch_size 2 \
#   --num_train_epochs 1 \
#   --warmup_steps 0 \
#   --save_steps 1000 \
#   --bf16 \
#   --learning_rate 5e-5 \
#   --freeze_llm \
#   --freeze_audio_tower \
#   --freeze_audio_encoder_proj \
#   --freeze_embed \
#   --freeze_text_added_tokens \
#   --deepspeed_file deepspeed_files/ds_config_zero2.json \
#   --ta_rate 0.8

# torchrun \
#   ${HOST_NUM:+--nnodes ${HOST_NUM}} \
#   --nproc_per_node ${n_gpus_per_node} \
#   --node-rank ${node_rank} \
#   ${CHIEF_IP:+--master-addr ${CHIEF_IP}} \
#   --master_port=${port} \
#   --local-addr ${LOCAL_IP} \
#   trainer/trainer_ddp_multi_round.py \
#   --model_path output_train/paper_model/checkpoint-320000 \
#   --audio_tokenizer_path higgs-audio-v2-tokenizer \
#   --train_data_dir higgs_training_data_mini/ \
#   --task_type multi_speaker_chat \
#   --step 3 \
#   --output_dir output_train/step_3_1220 \
#   --per_device_train_batch_size 2 \
#   --num_train_epochs 1 \
#   --warmup_steps 0 \
#   --save_steps 1000 \
#   --bf16 \
#   --learning_rate 5e-5 \
#   --freeze_llm \
#   --freeze_audio_tower \
#   --freeze_audio_encoder_proj \
#   --freeze_embed \
#   --freeze_text_added_tokens \
#   --deepspeed_file deepspeed_files/ds_config_zero2.json \
#   --ta_rate 0.8

torchrun \
  ${HOST_NUM:+--nnodes ${HOST_NUM}} \
  --nproc_per_node ${n_gpus_per_node} \
  --node-rank ${node_rank} \
  ${CHIEF_IP:+--master-addr ${CHIEF_IP}} \
  --master_port=${port} \
  --local-addr ${LOCAL_IP} \
  trainer/trainer_ddp_multi_round.py \
  --model_path output_train/step_3_1220/checkpoint-17965 \
  --audio_tokenizer_path higgs-audio-v2-tokenizer \
  --train_data_dir higgs_training_data_mini/ \
  --task_type multi_speaker_chat \
  --step 4 \
  --output_dir output_train/step_4_1224 \
  --per_device_train_batch_size 2 \
  --num_train_epochs 2 \
  --warmup_steps 0 \
  --save_steps 1000 \
  --bf16 \
  --learning_rate 5e-5 \
  --freeze_llm \
  --freeze_audio_tower \
  --freeze_audio_encoder_proj \
  --freeze_embed \
  --freeze_text_added_tokens \
  --deepspeed_file deepspeed_files/ds_config_zero2.json \
  --ta_rate 0.8

# torchrun \
#   ${HOST_NUM:+--nnodes ${HOST_NUM}} \
#   --nproc_per_node ${n_gpus_per_node} \
#   --node-rank ${node_rank} \
#   ${CHIEF_IP:+--master-addr ${CHIEF_IP}} \
#   --master_port=${port} \
#   --local-addr ${LOCAL_IP} \
#   trainer/trainer_ddp_multi_round.py \
#   --model_path higgs-audio-v2-generation-3B-base \
#   --audio_tokenizer_path higgs-audio-v2-tokenizer \
#   --train_data_dir higgs_training_data_mini/ \
#   --task_type multi_speaker_chat \
#   --output_dir output_train/1016_interval_unfreeze \
#   --per_device_train_batch_size 2 \
#   --num_train_epochs 3 \
#   --save_steps 1000 \
#   --bf16 \
#   --model_init \
#   --freeze_audio_tower \
#   --ta_rate 0.8

# #  no   --freeze_text_added_tokens \


# torchrun \
#   ${HOST_NUM:+--nnodes ${HOST_NUM}} \
#   --nproc_per_node ${n_gpus_per_node} \
#   --node-rank ${node_rank} \
#   ${CHIEF_IP:+--master-addr ${CHIEF_IP}} \
#   --master_port=${port} \
#   --local-addr ${LOCAL_IP} \
#   trainer/trainer_ddp_multi_round.py \
#   --model_path output_train/0916_without_audio_in_token_llama_aligner_finetune/checkpoint-81795 \
#   --audio_tokenizer_path higgs-audio-v2-tokenizer \
#   --train_data_dir higgs_training_data_mini/ \
#   --task_type multi_speaker_chat \
#   --output_dir output_train/0919_without_audio_in_token_llaso_tts_finetune \
#   --per_device_train_batch_size 2 \
#   --num_train_epochs 3 \
#   --save_steps 1000 \
#   --bf16 \
#   --freeze_audio_tower \
#   --ta_rate 0.8

#  no   --freeze_text_added_tokens \










# torchrun --nproc_per_node=8 --master_port 29909 trainer/trainer_ddp_multi_round.py \
  # --model_path higgs-audio-v2-generation-3B-base \
#   --audio_tokenizer_path higgs-audio-v2-tokenizer \
#   --train_data_dir higgs_training_data_mini/ \
#   --task_type multi_speaker_chat \
#   --output_dir output_train/0910_multi_round_one_epoch \
#   --per_device_train_batch_size 2 \
#   --num_train_epochs 1 \
#   --save_steps 2000 \
#   --bf16 \
#   --freeze_audio_tower
