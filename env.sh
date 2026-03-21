export http_proxy=http://star-proxy.oa.com:3128 
export https_proxy=http://star-proxy.oa.com:3128
pip config set global.index-url https://pypi.org/simple
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r /apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/higgs/train-higgs-audio/require.txt
conda install ffmpeg -c pytorch -y
python load.py
