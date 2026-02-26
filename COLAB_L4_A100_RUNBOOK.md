# Colab L4/A100 Runbook (No Heredocs, GPU Required)

Use this for a fresh Google Colab session where `nvidia-smi` shows `L4` or `A100`.
This flow avoids heredoc cells and keeps commands copy/paste-safe.

## 1) Runtime

- `Runtime -> Change runtime type -> GPU`
- Restart runtime if prompted.

## 2) Upload and unpack project zip

```python
from google.colab import files
files.upload()  # choose: sam-audio-colab-upload.zip
```

```python
!unzip -q -o sam-audio-colab-upload.zip -d /content
!test -f /content/colab_smoke_test.py && echo "FOUND /content/colab_smoke_test.py" || echo "MISSING /content/colab_smoke_test.py"
```

## 3) Install system dependencies

```python
!apt-get -qq update
!apt-get -qq install -y python3-venv python3-pip-whl python3-setuptools-whl ffmpeg curl
```

## 4) Create venv (robust path for Colab Python 3.12)

```python
!rm -rf /content/venv-sam
!python3 -m venv --without-pip /content/venv-sam
!test -x /content/venv-sam/bin/python && echo "VENV_OK" || echo "VENV_MISSING"
```

```python
!curl -sS https://bootstrap.pypa.io/get-pip.py -o /content/get-pip.py
!/content/venv-sam/bin/python /content/get-pip.py
!/content/venv-sam/bin/python -m pip --version
```

## 5) Install Python dependencies into venv

```python
!/content/venv-sam/bin/python -m pip install -U pip setuptools wheel
!/content/venv-sam/bin/python -m pip install "numpy==1.26.4" "protobuf<3.20" soundfile torch torchdiffeq
!/content/venv-sam/bin/python -m pip install "huggingface-hub==0.34.6" "transformers==4.56.1"
!/content/venv-sam/bin/python -m pip install "sam_audio @ git+https://github.com/facebookresearch/sam-audio"
```

```python
!/content/venv-sam/bin/python -c "import numpy,torch,transformers,sam_audio; print('deps ok')"
```

## 6) Verify GPU from venv

```python
!/content/venv-sam/bin/python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count()); print('cuda_version=', torch.version.cuda); print('gpu_name=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
!nvidia-smi
```

## 7) Set Hugging Face token from Colab Secret

Create a Colab secret named `HF_TOKEN` first, then run:

```python
import os
from google.colab import userdata

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN").strip()
print("HF token loaded:", os.environ["HF_TOKEN"][:4] + "...")
```

## 8) Download model

```python
from huggingface_hub import login, snapshot_download
import os

login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
model_dir = snapshot_download("facebook/sam-audio-large-tv", token=os.environ["HF_TOKEN"])
print("MODEL_DIR =", model_dir)
```

```python
from pathlib import Path

model_root = Path("/root/.cache/huggingface/hub/models--facebook--sam-audio-large-tv/snapshots")
latest_model = str(sorted([p for p in model_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)[-1])
print("latest_model =", latest_model)
```

## 9) Upload input audio and create tiny test clip

```python
from google.colab import files
uploaded_audio = files.upload()
input_audio = next(iter(uploaded_audio.keys()))
print("input_audio =", input_audio)
```

```python
!ffmpeg -y -i "/content/$input_audio" -t 8 /content/tiny.wav
```

## 10) Run smoke test (conservative GPU settings)

```python
import os
import subprocess

env = os.environ.copy()
env["USE_TF"] = "0"
env["TRANSFORMERS_NO_TF"] = "1"
env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

cmd = [
    "/content/venv-sam/bin/python",
    "/content/colab_smoke_test.py",
    "--input", "/content/tiny.wav",
    "--model-dir", latest_model,
    "--description", "speech",
    "--output-dir", "/content/colab_output",
    "--device", "cuda",
    "--no-cpu-fallback",
    "--trial-seconds", "1",
    "--chunk-duration", "1",
    "--overlap", "0.0",
    "--rerank", "1",
    "--memory-fraction", "0.20",
]

p = subprocess.run(cmd, env=env, capture_output=True, text=True)
print("RC:", p.returncode)
print("STDOUT:\n", p.stdout[-12000:])
print("STDERR:\n", p.stderr[-12000:])
```

## 11) Download result (only if `RC: 0`)

```python
from google.colab import files
files.download("/content/colab_output/audio_cleanup_result.zip")
```

## 12) Quick failure guide

- `RC = -9`: process killed by runtime memory pressure. Retry on A100 if on L4, and keep conservative flags.
- `ModuleNotFoundError`: venv install step did not complete; rerun steps 4-5.
- `KeyError: 'HF_TOKEN'`: set Colab secret `HF_TOKEN`, then rerun step 7.
- `CalledProcessError` with little detail: run step 10 exactly as written (it captures stdout/stderr).

