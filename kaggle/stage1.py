import os, subprocess, sys, json, glob
os.makedirs("/kaggle/working/results", exist_ok=True); os.chdir("/kaggle/working")
def run(cmd):
    print("$", cmd, flush=True); p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(p.stdout[-2500:], p.stderr[-1200:] if p.returncode else "", "rc", p.returncode, flush=True)
run("pip install -q -U transformers flash-linear-attention nvidia-ml-py 2>&1 | tail -1")
run("nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv; pip list 2>/dev/null | grep -iE '^(torch|transformers|flash-linear-attention|triton) '")
for tag, m in [("qwen4b", "Qwen/Qwen3.5-4B"), ("gemma2b", "google/gemma-4-E2B-it"), ("sarvam", "sarvamai/sarvam-1")]:
    run(f"cd src && python m2_pilot.py --model {m} --n 20 --repeats 3 --langs eng_Latn,hin_Deva,tam_Taml --out ../results/m2_pilot_{tag}.json 2>&1 | grep -vE 'Loading weights|warn'")
run("cd src && python m4_spec.py --model Qwen/Qwen3.5-4B --draft Qwen/Qwen3.5-0.8B --n 3 --out ../results/m4_smoke_qwen.json 2>&1 | grep -vE 'Loading weights|warn' | tail -25")
