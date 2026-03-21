export http_proxy=http://star-proxy.oa.com:3128 
export https_proxy=http://star-proxy.oa.com:3128
export http_proxy="http://9.21.0.122:11113"
export https_proxy="http://9.21.0.122:11113"

export PYTHONPATH=src:$PYTHONPATH
export NCCL_LAUNCH_MODE=GROUP

python eval_seedtts_testset.py --eval_task asr --lang en --gen_wav_dir "" --gpu_nums 8
