# # export http_proxy=http://star-proxy.oa.com:3128 
# # export https_proxy=http://star-proxy.oa.com:3128
export http_proxy="http://9.21.0.122:11113"
export https_proxy="http://9.21.0.122:11113"
# set -x
# export CUDA_HOME=/usr/local/cuda-12.4/
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:LD_LIBRARY_PATH$


# NET_TYPE="low"
# export OMP_NUM_THREADS=8
# export NCCL_IB_TIMEOUT=24


# # export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG=INFO
# # export NCCL_DEBUG_SUBSYS=ALL
# if [[ "${NET_TYPE}" = "low" ]]; then
#     export NCCL_SOCKET_IFNAME=eth1
#     export NCCL_IB_DISABLE=1
#     export NCCL_IB_GID_INDEX=3
#     nccl_ib_hca=$(bash show_gids |  grep $(hostname -I) |  grep v2 | awk '{print $1 ":" $2}' )

#     export NCCL_IB_HCA=mlx5_2:1,mlx5_2:1
#     export NCCL_IB_SL=3
#     export NCCL_CHECK_DISABLE=1
#     export NCCL_P2P_DISABLE=1
#     export NCCL_LL_THRESHOLD=16384
#     export NCCL_IB_CUDA_SUPPORT=1
# else
#     export NCCL_IB_GID_INDEX=3
#     export NCCL_IB_SL=3
#     export NCCL_CHECK_DISABLE=1
#     export NCCL_P2P_DISABLE=0
#     export NCCL_IB_DISABLE=0
#     export NCCL_LL_THRESHOLD=16384
#     export NCCL_IB_CUDA_SUPPORT=1
#     export NCCL_SOCKET_IFNAME=bond1
#     export UCX_NET_DEVICES=bond1
#     export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
#     export NCCL_COLLNET_ENABLE=0
#     export SHARP_COLL_ENABLE_SAT=0
#     export NCCL_NET_GDR_LEVEL=2
#     export NCCL_IB_QPS_PER_CONNECTION=4
#     export NCCL_IB_TC=160
#     export NCCL_PXN_DISABLE=1
# fi
# node_num=${HOST_NUM}
# n_gpus_per_node=${HOST_GPU_NUM}
# node_rank=${INDEX:-0}
# n_gpu=${HOST_GPU_NUM:-1}
# port=2348
# n_gpus=$((node_num * n_gpus_per_node))

# cat /etc/hosts
# echo "ngpus: ${n_gpus}"
# echo "Hosts: ${HOST_NUM}"
# echo "node_rank: ${node_rank}"
# echo "master: ${CHIEF_IP}:${port}"
# echo "local ip: ${LOCAL_IP}"   
# echo "start infer..."


export PYTHONPATH=src:$PYTHONPATH


accelerate launch --multi-gpu --main_process_port=29501 \
    examples/serve_engine/run_test_multi_GPUs_async.py

# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port=29501 \
#     examples/serve_engine/run_test_multi_GPUs_async.py
