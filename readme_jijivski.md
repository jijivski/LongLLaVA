

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


clear
export PYTHONPATH=/nesa_data/remote_shome/zch/workspace/LongLLaVA/:$PYTHONPATH
conda activate LongLLaVa
time bash Eval.sh

CUDA_VISIBLE_DEVICES="6,7" time bash Eval.sh
export CUDA_VISIBLE_DEVICES=6,7

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.
pip show numpy : 2.0.2
then downgrade,
pip install "numpy<2.0"

(LongLLaVa) (base) root@a100-1:/nesa_data/remote_shome/zch/workspace/LongLLaVA# pip install numpy<2
bash: 2: No such file or directory
scipy 1.9.1 requires numpy<1.25.0,>=1.18.5, but you have numpy 1.26.4 which is incompatible.
Successfully installed numpy-1.26.4

/wangbenyou/xidong/VisionJamba/benchmarks/VIAH/model_viah_qa.py
视频的



(LongLLaVa) root@a100-1:/nesa_data/remote_shome/zch/workspace/LongLLaVA# grep -r "guiming"
ckpts/10SFT2dSenseLong176K copy/config.json:  "mm_vision_tower": "/wangbenyou/guimingchen/models/clip_vit_large_patch14_336",
ckpts/10SFT2dSenseLong176K/config.json:  "mm_vision_tower": "/wangbenyou/guimingchen/models/clip_vit_large_patch14_336",
benchmarks/POPE/eval_pope.sh:        --image-folder /wangbenyou/guimingchen/datasets/MSCOCO/val2014 \


https://github.com/state-spaces/mamba

pip install mamba-ssm[causal-conv1d] --no-build-isolation


pip install /nesa_data/remote_shome/zch/workspace/LongLLaVA/addition/mamba_ssm-2.2.2+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu124torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

cd /nesa_data/remote_shome/zch/workspace/LongLLaVA/addition/build
git clone https://github.com/state-spaces/mamba.git
git clone https://github.com/Dao-AILab/causal-conv1d.git

sudo apt-get install build-essential python3-dev
<!-- (LongLLaVa) (base) root@a100-1:/nesa_data/remote_shome/zch/workspace/LongLLaVA/addition/build# sudo apt-get install build-essential python3-dev
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
build-essential is already the newest version (12.9ubuntu3).
python3-dev is already the newest version (3.10.6-1~22.04.1).
python3-dev set to manually installed.
The following packages were automatically installed and are no longer required:
  nvidia-firmware-560-560.35.03 nvidia-modprobe
Use 'sudo apt autoremove' to remove them.
0 upgraded, 0 newly installed, 0 to remove and 27 not upgraded. -->
cd causal-conv1d
python setup.py build

cd ../mamba
python setup.py build



cp -r build/lib.linux-x86_64-cpython-310/* /nesa_data/remote_shome/xianfeng/anaconda3/envs/LongLLaVa/lib/python3.10/site-packages/
cd ../causal-conv1d
cp -r build/lib.linux-x86_64-cpython-310/* /nesa_data/remote_shome/xianfeng/anaconda3/envs/LongLLaVa/lib/python3.10/site-packages/


当然,我很乐意为您详细解释这个解决方案。这是一个从源代码编译和安装的过程,通常在预编译的wheel文件不兼容时使用。让我们逐步分析:

1. 获取源代码:
   首先,您需要从GitHub获取causal-conv1d和mamba的源代码。
   ```
   git clone https://github.com/state-spaces/mamba.git
   git clone https://github.com/Dao-AILab/causal-conv1d.git
   ```

2. 准备编译环境:
   确保您有适当的编译工具和依赖项。这通常包括C++编译器、CUDA工具包和Python开发头文件。
   ```
   sudo apt-get install build-essential python3-dev
   ```

3. 编译causal-conv1d:
   进入causal-conv1d目录并编译:
   ```
   cd causal-conv1d
   python setup.py build
   ```

4. 编译mamba:
   同样,进入mamba目录并编译:
   ```
   cd ../mamba
   python setup.py build
   ```

5. 复制编译结果:
   编译完成后,生成的文件通常位于`build/lib.linux-x86_64-cpython-310/`目录下。这里的`310`表示Python 3.10版本,可能会根据您的Python版本有所不同。

6. 安装到虚拟环境:
   假设您的虚拟环境位于`../venv/`,您需要将编译好的文件复制到虚拟环境的site-packages目录:
   ```
   cp -r build/lib.linux-x86_64-cpython-310/* ../venv/lib/python3.10/site-packages/
   ```
   对causal-conv1d和mamba都执行这个步骤。

7. 更新PYTHONPATH:
   确保您的PYTHONPATH包含了新安装的库的路径:
   ```
   export PYTHONPATH=$PYTHONPATH:../venv/lib/python3.10/site-packages/
   ```

这个方法的优点是:
- 可以确保库与您的系统和CUDA版本完全兼容。
- 允许您对源代码进行自定义修改(如果需要的话)。
- 解决了预编译wheel文件可能存在的兼容性问题。

注意事项:
- 编译过程可能需要一些时间,特别是对于大型项目。
- 确保您有足够的磁盘空间用于编译。
- 如果遇到编译错误,可能需要安装额外的依赖或解决版本冲突。

这个方法虽然比直接安装wheel文件更复杂,但它提供了更大的灵活性和兼容性。在处理特定的CUDA版本或系统配置时,这种方法特别有用。


succeed!

python 
>>>
from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
>>>



python ./benchmarks/vstarbench/generate_score.py --output_path ./benchmarks/vstarbench/outputs/001_norm_336 --score_path ./benchmarks/vstarbench/outputs/001_norm_336/score.json