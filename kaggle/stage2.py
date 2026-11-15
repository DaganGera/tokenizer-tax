import os, subprocess
os.makedirs("/kaggle/working/results", exist_ok=True); os.chdir("/kaggle/working")
def run(cmd):
    print("$", cmd, flush=True); p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(p.stdout[-3000:], p.stderr[-800:] if p.returncode else "", "rc", p.returncode, flush=True)
run("pip install -q -U transformers flash-linear-attention nvidia-ml-py 2>&1 | tail -1")
F = "grep -vE 'Loading weights|warn|Fetching|unauthenticated|max_length|generation_config|generation flags'"
run(f"cd src && python m4_spec.py --model google/gemma-4-E2B-it --draft google/gemma-4-E2B-it-assistant --n 2 --out ../results/m4_smoke_gemma.json 2>&1 | {F} | tail -25")
run(f"cd src && python m4_spec.py --model Qwen/Qwen3-4B --draft Qwen/Qwen3-0.6B --n 2 --out ../results/m4_smoke_qwen3.json 2>&1 | {F} | tail -15")
