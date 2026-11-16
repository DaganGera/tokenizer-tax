"""M4 analysis: per method energy/latency/acceptance, exactness, and whether the Indian-language/English energy gap shrinks."""
import json, sys, glob, numpy as np
INDIC = "hin_Deva ben_Beng tam_Taml tel_Telu mar_Deva guj_Gujr kan_Knda mal_Mlym pan_Guru ory_Orya".split()

def gap(J):          # J[method][lang] -> array over sentences; gap = mean over Indic langs of mean J / mean J(English)
    return np.mean([J[l].mean() for l in INDIC if l in J]) / J["eng_Latn"].mean()

def analyse(path, n_boot=2000, clean=False):
    d = json.load(open(path)); rows = d["rows"]
    if clean:      # sensitivity: keep sentences whose BASELINE output ended naturally (3..159 new tokens)
        ok = {(r["lang"], r["sent"]) for r in rows if r["method"] == "baseline" and 3 <= r["new_tokens"] < 160}
        rows = [r for r in rows if (r["lang"], r["sent"]) in ok]
    methods = sorted({r["method"] for r in rows}); out = {"clean_subset": clean, "n_sentences_kept": len({(r["lang"], r["sent"]) for r in rows}), "model": d["model"], "draft": d.get("draft"), "draft_params": d.get("draft_params"), "unsupported": d.get("unsupported"), "methods": {}}
    J = {m: {} for m in methods}
    for m in methods:
        for l in sorted({r["lang"] for r in rows}):
            J[m][l] = np.array([r["joules"] for r in sorted(rows, key=lambda r: r["sent"]) if r["method"] == m and r["lang"] == l])
    rng = np.random.default_rng(0)
    def boot_gap(m, base):
        res = []
        for _ in range(n_boot):
            idx = {l: rng.integers(0, len(J[m][l]), len(J[m][l])) for l in J[m]}
            gm = gap({l: J[m][l][idx[l]] for l in J[m]}); gb = gap({l: J[base][l][idx[l]] for l in J[base]})
            res.append(100 * (gb - gm) / gb)
        return [float(np.percentile(res, 2.5)), float(np.percentile(res, 97.5))]
    for m in methods:
        rs = [r for r in rows if r["method"] == m]; per = {}
        for l in J[m]:
            x = [r for r in rs if r["lang"] == l]
            per[l] = dict(mean_j=J[m][l].mean(), mean_s=np.mean([r["seconds"] for r in x]), new_tokens=np.mean([r["new_tokens"] for r in x]),
                          acceptance=sum(r["accepted"] for r in x) / max(1, sum(r["proposed"] for r in x)), identical=sum(r["identical_to_baseline"] for r in x) / len(x))
        o = dict(per_lang=per, gap=gap(J[m]), identical_total=f"{sum(r['identical_to_baseline'] for r in rs)}/{len(rs)}",
                 total_j=float(sum(J[m][l].sum() for l in J[m])))
        if m != "baseline" and "baseline" in J:
            o["gap_reduction_pct"] = 100 * (gap(J["baseline"]) - o["gap"]) / gap(J["baseline"]); o["gap_reduction_ci95"] = boot_gap(m, "baseline")
            o["energy_change_pct_indic"] = 100 * (np.mean([J[m][l].mean() for l in INDIC]) / np.mean([J["baseline"][l].mean() for l in INDIC]) - 1)
            o["energy_change_pct_english"] = 100 * (J[m]["eng_Latn"].mean() / J["baseline"]["eng_Latn"].mean() - 1)
        out["methods"][m] = o
    return out

if __name__ == "__main__":
    res = [analyse(p, clean=c) for c in (False, True) for p in [f for f in sorted(glob.glob(sys.argv[1])) if "INVALID" not in f and "smoke" not in f]]; json.dump(res, open(sys.argv[2], "w"), indent=1, default=float)
    for o in res:
        print("CLEAN SUBSET" if o["clean_subset"] else "ALL", o["n_sentences_kept"], "sentences |", o["model"], "draft", o["draft"], o["draft_params"], "unsupported", o["unsupported"])
        for m, v in o["methods"].items():
            acc = np.mean([x["acceptance"] for x in v["per_lang"].values()]); extra = f" gap_red={v['gap_reduction_pct']:.1f}% CI{[round(c,1) for c in v['gap_reduction_ci95']]} dE_indic={v['energy_change_pct_indic']:+.1f}% dE_eng={v['energy_change_pct_english']:+.1f}%" if "gap_reduction_pct" in v else ""
            print(f"  {m:14} gap={v['gap']:.3f} identical={v['identical_total']} acc={acc:.3f}{extra}")
