# Colab Fresh Runbook (Venv Path, Recommended)

This avoids Colab system-package conflicts by running everything inside `/content/venv-sam`.
Use a **fresh Colab session** and run cells in order.

## 0) Optional runtime setting
- `Runtime -> Change runtime type`
- If GPU is unavailable, continue in CPU mode.

## 1) Upload and unpack project zip
```python
from google.colab import files
files.upload()  # choose: sam-audio-colab-upload.zip
```

```python
!unzip -q -o sam-audio-colab-upload.zip -d /content
!test -f /content/colab_smoke_test.py && echo "FOUND /content/colab_smoke_test.py" || echo "MISSING"
```

## 2) Create virtual environment + install deps
```python
!apt-get -qq update
!apt-get -qq install -y ffmpeg python3-venv
!python3 -m venv /content/venv-sam
```

```python
!/content/venv-sam/bin/python -m pip install --upgrade pip setuptools wheel
!/content/venv-sam/bin/pip install "numpy==1.26.4" "protobuf<3.20" soundfile torch torchdiffeq
!/content/venv-sam/bin/pip install "huggingface-hub==0.34.6" "transformers==4.56.1"
!/content/venv-sam/bin/pip install "sam_audio @ git+https://github.com/facebookresearch/sam-audio"
```

## 3) Verify venv imports
```python
!/content/venv-sam/bin/python - <<'PY'
import numpy, huggingface_hub, transformers, sam_audio
print('numpy', numpy.__version__)
print('hub', huggingface_hub.__version__)
print('transformers', transformers.__version__)
print('sam_audio import ok')
PY
```

## 4) Patch Python 3.12 compatibility into smoke-test script
```python
%%bash
python - <<'PY'
from pathlib import Path
p = Path('/content/colab_smoke_test.py')
s = p.read_text()
shim = '''import pkgutil
import importlib.machinery

if not hasattr(pkgutil, "ImpImporter"):
    pkgutil.ImpImporter = pkgutil.zipimporter
if not hasattr(pkgutil, "ImpLoader"):
    pkgutil.ImpLoader = pkgutil.zipimporter
if not hasattr(importlib.machinery.FileFinder, "find_module"):
    def _find_module(self, fullname):
        spec = self.find_spec(fullname)
        return None if spec is None else spec.loader
    importlib.machinery.FileFinder.find_module = _find_module

'''
if 'FileFinder.find_module' not in s:
    s = s.replace('from __future__ import annotations\n\n', 'from __future__ import annotations\n\n' + shim)
    p.write_text(s)
print('patched:', p)
PY
```

## 5) Authenticate HF + download model (inside venv)
With Colab secret `HF_TOKEN`:

```python
from google.colab import userdata
HF_TOKEN = userdata.get('HF_TOKEN').strip()
print('token prefix:', HF_TOKEN[:3])
```

```python
!/content/venv-sam/bin/python - <<'PY'
import os
from huggingface_hub import login, snapshot_download
HF_TOKEN = os.environ['HF_TOKEN']
login(token=HF_TOKEN, add_to_git_credential=False)
model_dir = snapshot_download('facebook/sam-audio-large-tv', token=HF_TOKEN)
print(model_dir)
PY
```

If you use secret, export it first:

```python
import os
from google.colab import userdata
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN').strip()
```

## 6) Upload audio
```python
from google.colab import files
uploaded_audio = files.upload()
input_audio = next(iter(uploaded_audio.keys()))
print('input_audio =', input_audio)
```

## 7) Create tiny clip (faster)
```python
!ffmpeg -y -i "/content/$input_audio" -t 8 /content/tiny.wav
```

## 8) Run smoke test using venv Python (CPU)
```python
!USE_TF=0 TRANSFORMERS_NO_TF=1 /content/venv-sam/bin/python /content/colab_smoke_test.py \
  --input /content/tiny.wav \
  --model-dir /root/.cache/huggingface/hub/models--facebook--sam-audio-large-tv/snapshots/b9fc94687ec044c570cbe30b8c28100cd056f1cb \
  --description "speech" \
  --output-dir /content/colab_output \
  --device cpu \
  --chunk-duration 4 \
  --overlap 0.25 \
  --rerank 1 \
  --memory-fraction 0.6
```

If your snapshot hash differs, use dynamic path:

```python
!USE_TF=0 TRANSFORMERS_NO_TF=1 /content/venv-sam/bin/python - <<'PY'
from pathlib import Path
import subprocess
model_root = Path('/root/.cache/huggingface/hub/models--facebook--sam-audio-large-tv/snapshots')
latest = sorted([p for p in model_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)[-1]
cmd = [
    '/content/venv-sam/bin/python', '/content/colab_smoke_test.py',
    '--input', '/content/tiny.wav',
    '--model-dir', str(latest),
    '--description', 'speech',
    '--output-dir', '/content/colab_output',
    '--device', 'cpu',
    '--chunk-duration', '4',
    '--overlap', '0.25',
    '--rerank', '1',
    '--memory-fraction', '0.6',
]
print('Running with model:', latest)
subprocess.run(cmd, check=True)
PY
```

## 9) Download output
```python
from google.colab import files
files.download('/content/colab_output/audio_cleanup_result.zip')
```

## 10) If foreground run gets interrupted, run background
```python
!pkill -f colab_smoke_test.py || true
!nohup env USE_TF=0 TRANSFORMERS_NO_TF=1 PYTHONUNBUFFERED=1 /content/venv-sam/bin/python /content/colab_smoke_test.py \
  --input /content/tiny.wav \
  --model-dir /root/.cache/huggingface/hub/models--facebook--sam-audio-large-tv/snapshots/b9fc94687ec044c570cbe30b8c28100cd056f1cb \
  --description "speech" \
  --output-dir /content/colab_output \
  --device cpu \
  --chunk-duration 4 \
  --overlap 0.25 \
  --rerank 1 \
  --memory-fraction 0.6 \
  > /content/run.log 2>&1 &
```

```python
!ps -ef | grep colab_smoke_test.py | grep -v grep
!tail -n 120 /content/run.log
!ls -lah /content/colab_output
```
