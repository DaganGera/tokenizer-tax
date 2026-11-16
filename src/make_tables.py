"""Markdown tables for the README, generated from results/*.json (no hand-typed numbers)."""
import json
I = "hin_Deva ben_Beng tam_Taml tel_Telu mar_Deva guj_Gujr kan_Knda mal_Mlym pan_Guru ory_Orya".split()
m1 = json.load(open("results/m1_token_counts.json"))["results"]; m3 = {o["model"].split("/")[-1]: o for o in json.load(open("results/analysis_m3.json"))}
m4 = json.load(open("results/analysis_m4.json")); out = []
out.append("### Table 1. Tokens vs English (M1, same 200 sentences)\n\n| language | " + " | ".join(m1) + " |\n|---|" + "---|" * len(m1))
for l in I: out.append(f"| {l[:3]} | " + " | ".join(f"{m1[t]['per_lang'][l]['ratio_to_english']:.2f}" for t in m1) + " |")
out.append("| langs under 1.5x | " + " | ".join(f"{sum(m1[t]['per_lang'][l]['ratio_to_english'] < 1.5 for l in I)}/10" for t in m1) + " |")
out.append("\n### Table 2. Energy per sentence, ratio to English (M3, T4, batch 1, raw joules; token ratio in brackets)\n\n| language | " + " | ".join(m3) + " |\n|---|" + "---|" * len(m3))
for l in I: out.append(f"| {l[:3]} | " + " | ".join(f"{m3[m]['langs'][l]['energy_ratio']:.2f}x ({m3[m]['langs'][l]['token_ratio']:.2f}x)" for m in m3) + " |")
out.append("| English J/sentence | " + " | ".join(f"{m3[m]['langs']['eng_Latn']['mean_j']:.1f} J" for m in m3) + " |")
out.append("\n### Table 3. Is joules per token constant? (M3)\n\n| model | parameters | slope J/token | R² | J/token vs English, range over 10 languages | ms/token | bandwidth floor ms (assumption) |\n|---|---|---|---|---|---|---|")
for m, o in m3.items():
    dev = [o["langs"][l]["jpt_vs_english_pct"] for l in I]; ms = sum(v["mean_s"] / v["mean_tokens"] for v in o["langs"].values()) / len(o["langs"]) * 1000
    out.append(f"| {m} | {o['params']/1e9:.2f}B | {o['fit_all']['slope_j_per_token']:.3f} | {o['fit_all']['r2']:.4f} | {min(dev):+.1f}% to {max(dev):+.1f}% | {ms:.1f} | {o['params']*2/320e9*1000:.1f} |")
for tag, flag in (("all 220 sentences (pre-registered analysis)", False), ("sensitivity: only sentences whose baseline output ended naturally", True)):
    out.append(f"\n### Table 4{'b' if flag else 'a'}. Speculative decoding, {tag}\n\n| model | method | sentences | acceptance | J/sentence change, Indian | J/sentence change, English | gap reduction % (95% CI) | outputs identical to baseline |\n|---|---|---|---|---|---|---|---|")
    for o in [x for x in m4 if x["clean_subset"] == flag]:
        for m, v in o["methods"].items():
            if m == "baseline": continue
            acc = sum(v["per_lang"][l]["acceptance"] for l in I) / 10
            out.append(f"| {o['model'].split('/')[-1]} | {m} | {o['n_sentences_kept']} | {acc:.2f} | {v['energy_change_pct_indic']:+.0f}% | {v['energy_change_pct_english']:+.0f}% | {v['gap_reduction_pct']:+.1f} ({v['gap_reduction_ci95'][0]:+.1f} to {v['gap_reduction_ci95'][1]:+.1f}) | {v['identical_total']} |")
open("results/tables.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
