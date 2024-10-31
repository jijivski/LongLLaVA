

conda create -n LongLLaVa
conda env remove --name LongLLaVa
conda create --name LongLLaVa --clone mar



conda activate LongLLaVa
export PIP_INDEX_URL= https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_INDEX_URL=https://pypi.doubanio.com/simple
pip install torch -i https://pypi.douban.com/simple


pip install -r requirements.txt




主要是clone省略了大部分的时间
设置的index似乎生效的

w-hotfix-0.6 pycountry-24.6.1 pycparser-2.22 pydantic-2.9.2 pydantic-core-2.23.4 python-dotenv-1.0.1 pyzmq-26.2.0 ray-2.37.0 referencing-0.35.1 regex-2023.10.3 rpds-py-0.20.0 safetensors-0.4.2 sentencepiece-0.2.0 sentry-sdk-2.17.0 setproctitle-1.3.3 smmap-5.0.1 sniffio-1.3.1 soundfile-0.12.1 soxr-0.5.0.post1 starlette-0.40.0 tiktoken-0.6.0 tokenizers-0.19.1 torch-2.4.0 torchaudio-2.4.0 torchdata-0.7.1 torchvision-0.19.0 tqdm-4.66.1 transformers-4.44.2 triton-3.0.0 typing-extensions-4.12.2 uvicorn-0.32.0 uvloop-0.21.0 vllm-0.5.5 vllm-flash-attn-2.6.1 waitress-3.0.0 wandb-0.17.3 watchfiles-0.24.0 websockets-13.1 xformers-0.0.27.post2 xxhash-3.5.0 yarl-1.16.0
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable.It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.

但是版本卡的很死...

export PIP_INDEX_URL=https://pypi.doubanio.com/simple
(LongLLaVa) root@a100-1:/nesa_data/remote_shome/zch/workspace/LongLLaVA# pip install -r requirements.txt
Looking in indexes: https://pypi.doubanio.com/simple
Collecting causal-conv1d@ git+https://github.com/Dao-AILab/causal-conv1d@f8c246707eff3431b5cf2ce6defa4b7f309ea9fb (from -r requirements.txt (line 3))
  Cloning https://github.com/Dao-AILab/causal-conv1d (to revision f8c246707eff3431b5cf2ce6defa4b7f309ea9fb) to /tmp/pip-install-_m63t656/causal-conv1d_a5a3332dc289462984f7f6129ee30b71
  Running command git clone --filter=blob:none --quiet https://github.com/Dao-AILab/causal-conv1d /tmp/pip-install-_m63t656/causal-conv1d_a5a3332dc289462984f7f6129ee30b71
  Running command git rev-parse -q --verify 'sha^f8c246707eff3431b5cf2ce6defa4b7f309ea9fb'
  Running command git fetch -q https://github.com/Dao-AILab/causal-conv1d f8c246707eff3431b5cf2ce6defa4b7f309ea9fb
  Resolved https://github.com/Dao-AILab/causal-conv1d to commit f8c246707eff3431b5cf2ce6defa4b7f309ea9fb
  Preparing metadata (setup.py) ... done
Collecting accelerate==0.33.0 (from -r requirements.txt (line 1))
  Downloading https://mirrors.cloud.tencent.com/pypi/packages/15/33/b6b4ad5efa8b9f4275d4ed17ff8a44c97276171341ba565fdffb0e3dc5e8/accelerate-0.33.0-py3-none-any.whl (315 kB)
Collecting deepspeed==0.14.4 (from -r requirements.txt (line 2))
  Downloading https://mirrors.cloud.tencent.com/pypi/packages/2e/06/7315113506f1804b8ba4f77cb31905c62f7080452a0af2d13eaadaa83a08/deepspeed-0.14.4.tar.gz (1.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.3/1.3 MB 524.5 kB/s eta 0:00:00
  Preparing metadata (setup.py) ... done
Collecting mamba-ssm==2.2.2 (from -r requirements.txt (line 4))
  Downloading https://mirrors.cloud.tencent.com/pypi/packages/f7/80/69bc14816fda4b30be3ac724e3efc713969dd545cd0bcb35abee6b8dfbf9/mamba_ssm-2.2.2.tar.gz (85 kB)


export PYTHONPATH=/nesa_data/remote_shome/zch/workspace/LongLLaVA/:$PYTHONPATH
conda activate LongLLaVa
CUDA_VISIBLE_DEVICES="6,7" time bash Eval.sh
export CUDA_VISIBLE_DEVICES=6,7
time bash Eval.sh



/wangbenyou/xidong/VisionJamba/benchmarks/VIAH/model_viah_qa.py
视频的