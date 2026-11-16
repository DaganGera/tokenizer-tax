"""Build results/summary.json from measured analysis files only. Nothing here is typed in by hand."""
import json

m1 = json.load(open("results/m1_token_counts.json"))
m3 = json.load(open("results/analysis_m3.json")); m4 = json.load(open("results/analysis_m4.json"))
meta = json.load(open("results/t4/m3_qwen35_4b.json"))["meter"]            # GPU info logged by the harness during a measured run

best = max(((v["energy_ratio"], l, o["model"]) for o in m3 for l, v in o["langs"].items() if l != "eng_Latn"))
g = next(o for o in m4 if o["draft"] and "gemma" in o["model"] and not o["clean_subset"]); d = g["methods"]["draft"]
gc = next(o for o in m4 if o["draft"] and "gemma" in o["model"] and o["clean_subset"])["methods"]["draft"]   # headline: draft-model speculation on Gemma 4 E2B
def kaggle_versions(log="kaggle/out1/notebook82ea1dd796.log"):      # versions printed by `pip list` inside a Kaggle T4 job
    out = "".join(e["data"] for e in json.load(open(log)) if e["stream_name"] == "stdout")
    return {l.split()[0]: l.split()[1] for l in out.splitlines() if l.split()[:1] and l.split()[0] in ("torch", "transformers", "flash-linear-attention", "triton")}

summary = {
    "num_languages": len(m1["results"]["qwen3.5"]["per_lang"]),          # includes English
    "num_tokenizers": len(m1["results"]),
    "max_energy_ratio_vs_english": {"ratio": best[0], "language": best[1], "model": best[2]},
    "energy_gap_reduction_pct": {"value": d["gap_reduction_pct"], "ci95": d["gap_reduction_ci95"], "clean_subset_value": gc["gap_reduction_pct"], "clean_subset_ci95": gc["gap_reduction_ci95"], "indic_energy_change_pct": d["energy_change_pct_indic"], "english_energy_change_pct": d["energy_change_pct_english"], "outputs_identical_to_baseline": g["methods"]["draft"]["identical_total"], "method": "draft", "model": g["model"],
                                 "note": "negative = gap widened; see energy_change_pct_* in results/analysis_m4.json before interpreting"},
    "draft_model_params": g["draft_params"],
    "hardware": {"gpu": meta["gpu"], "driver": meta["driver"], "energy_meter": meta["mode"], "gpus_visible_to_job": 2, "gpus_used": 1},
    "software": kaggle_versions() | {"note": "from pip list in the stage-1 Kaggle T4 pilot job (python 3.13 image); later runs executed pip install -U, so patch versions may differ"},
}
json.dump(summary, open("results/summary.json", "w"), indent=1, default=float); print(json.dumps(summary, indent=1, default=float))
