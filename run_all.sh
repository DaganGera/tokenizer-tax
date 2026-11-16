#!/usr/bin/env bash
# Reproduce the whole study on a Kaggle T4 notebook (GPU on, internet on). One command:  bash run_all.sh
# FLORES+ is gated: accept the terms at huggingface.co/datasets/openlanguagedata/flores_plus and set HF_TOKEN
# (Kaggle: Add-ons > Secrets), or attach a private dataset with /kaggle/input/flores-plus-indic-subset/flores_plus_subset.json.
# Approx. GPU time on a T4: 5.5 h (Gemma 4 + Sarvam-1 M3 and all M4) + 2.7 h (Qwen3.5-4B M3) = about 8 h.
set -euo pipefail
pip install -q -U transformers flash-linear-attention nvidia-ml-py tiktoken matplotlib scipy sentencepiece
mkdir -p results/t4; cd src
python m1_tokens.py                                   # M1, CPU only: token counts -> ../results/m1_token_counts.json
python plot_m1.py
for m in google/gemma-4-E2B-it sarvamai/sarvam-1 Qwen/Qwen3.5-4B; do   # M2 reliability: 3 languages x 20 sentences x 3 repeats
  python m2_pilot.py --model $m --n 20 --repeats 3 --langs eng_Latn,hin_Deva,tam_Taml --out ../results/t4/m2_pilot_$(basename $m).json
done
for m in google/gemma-4-E2B-it sarvamai/sarvam-1 Qwen/Qwen3.5-4B; do   # M3: 11 languages x 200 sentences, one pass, teacher-forced
  python m2_pilot.py --model $m --n 200 --repeats 1 --out ../results/t4/m3_$(basename $m).json
done
python m4_spec.py --model google/gemma-4-E2B-it --draft google/gemma-4-E2B-it-assistant --n 20 --out ../results/t4/m4_gemma4_e2b.json
python m4_spec.py --model Qwen/Qwen3-4B --draft Qwen/Qwen3-0.6B --n 20 --out ../results/t4/m4_qwen3_4b.json
python m4_spec.py --model sarvamai/sarvam-1 --n 20 --out ../results/t4/m4_sarvam1.json
cd ..
python src/analyze_m3.py "results/t4/m3_*.json" results/analysis_m3.json
python src/analyze_m4.py "results/t4/m4_*.json" results/analysis_m4.json
python src/make_tables.py > /dev/null
