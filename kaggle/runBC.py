import os, subprocess
os.makedirs("/kaggle/working/results", exist_ok=True); os.chdir("/kaggle/working")
def run(cmd):
    print("$", cmd, flush=True); p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(p.stdout[-3000:], p.stderr[-800:] if p.returncode else "", "rc", p.returncode, flush=True)
run("pip install -q -U transformers flash-linear-attention nvidia-ml-py 2>&1 | tail -1")
F = "grep -vE 'Loading weights|warn|Fetching|unauthenticated|max_length|generation_config|generation flags'"
# M3: teacher-forced decoding, 11 languages x 200 sentences, 1 pass
run(f"cd src && python m2_pilot.py --model google/gemma-4-E2B-it --n 200 --repeats 1 --out ../results/m3_gemma4_e2b.json 2>&1 | {F}")
run(f"cd src && python m2_pilot.py --model sarvamai/sarvam-1 --n 200 --repeats 1 --out ../results/m3_sarvam1.json 2>&1 | {F}")
# M4: real greedy translation, 20 sentences x 11 languages, methods in random order per sentence
run(f"cd src && python m4_spec.py --model google/gemma-4-E2B-it --draft google/gemma-4-E2B-it-assistant --n 20 --out ../results/m4_gemma4_e2b.json 2>&1 | {F} | tail -8")
run(f"cd src && python m4_spec.py --model sarvamai/sarvam-1 --n 20 --out ../results/m4_sarvam1.json 2>&1 | {F} | tail -8")
run(f"cd src && python m4_spec.py --model Qwen/Qwen3-4B --draft Qwen/Qwen3-0.6B --n 20 --out ../results/m4_qwen3_4b.json 2>&1 | {F} | tail -8")
run(f"cd src && python m4_spec.py --model Qwen/Qwen3.5-4B --n 20 --out ../results/m4_qwen35_4b.json 2>&1 | {F} | tail -6")
