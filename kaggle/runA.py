import os, subprocess
os.makedirs("/kaggle/working/results", exist_ok=True); os.chdir("/kaggle/working")
def run(cmd):
    print("$", cmd, flush=True); p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(p.stdout[-3000:], p.stderr[-800:] if p.returncode else "", "rc", p.returncode, flush=True)
run("pip install -q -U transformers flash-linear-attention nvidia-ml-py 2>&1 | tail -1")
run("nvidia-smi -q -d CLOCK,PERFORMANCE | head -40")
run("cd src && python m2_pilot.py --model Qwen/Qwen3.5-4B --n 200 --repeats 1 --out ../results/m3_qwen35_4b.json 2>&1 | grep -vE 'Loading weights|warn|Fetching|unauthenticated'")
