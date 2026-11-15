import os, subprocess, time, sys
def sh(c): print(subprocess.run(c, shell=True, capture_output=True, text=True).stdout[-1500:])
sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; pip list 2>/dev/null | grep -iE '^(torch|transformers|accelerate|nvidia-ml-py|pynvml) '")
sh("pip install -q -U transformers flash-linear-attention nvidia-ml-py 2>&1 | tail -2; pip list 2>/dev/null | grep -iE '^(transformers|flash-linear) '")
sys.path.insert(0, "/kaggle/working/src")
import torch, pynvml as N
N.nvmlInit(); h = N.nvmlDeviceGetHandleByIndex(0); print(N.nvmlDeviceGetName(h), "gpus:", torch.cuda.device_count())
x = torch.randn(4096, 4096, device="cuda", dtype=torch.half); torch.cuda.synchronize(); t0 = time.time(); v = []
while time.time() - t0 < 3: x @ x; v.append((time.time(), N.nvmlDeviceGetTotalEnergyConsumption(h)))
ch = [b[0] - a[0] for a, b in zip(v, v[1:]) if b[1] != a[1]]; print("counter updates:", len(ch), "median gap s:", sorted(ch)[len(ch)//2] if ch else None)
from data import load_sentences; print(load_sentences("ory_Orya")[0][:60])
from transformers import AutoConfig
for r in ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-0.8B", "google/gemma-4-E2B-it", "sarvamai/sarvam-1"]:
    try: print(r, AutoConfig.from_pretrained(r).model_type)
    except Exception as e: print(r, "FAIL", str(e)[:150])
