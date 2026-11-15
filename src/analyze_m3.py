"""M3 analysis: J/sentence & latency per language, ratio to English, J-vs-tokens fit (slope, R^2), constancy of J/token."""
import json, sys, glob, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def load(path):
    d = json.load(open(path)); by = {}
    for r in d["rows"]: by.setdefault((r["lang"], r["sent"]), []).append(r)
    rows = [dict(lang=l, sent=s, tokens=v[0]["tokens"], joules=np.median([x["joules"] for x in v]), net=np.median([x["joules_net"] for x in v]),
                 seconds=np.median([x["seconds"] for x in v])) for (l, s), v in by.items()]       # median over repeats per sentence
    return d, rows

def boot(f, n=2000, seed=0):
    r = np.random.default_rng(seed); return [float(np.percentile(f(r), q)) for q in (2.5, 97.5)]

def analyse(path):
    d, rows = load(path); langs = sorted({r["lang"] for r in rows}); out = {"model": d["model"], "params": d.get("params"), "prefill_only": d.get("prefill_only"), "langs": {}}
    g = lambda l, k: np.array([r[k] for r in rows if r["lang"] == l]); eng = {k: g("eng_Latn", k) for k in ("tokens", "joules", "net", "seconds")}
    jpt_e = eng["joules"].sum() / eng["tokens"].sum()
    for l in langs:
        t, j, n, s = g(l, "tokens"), g(l, "joules"), g(l, "net"), g(l, "seconds"); m = len(t)
        ratio = j.mean() / eng["joules"].mean()
        out["langs"][l] = dict(n=m, mean_tokens=t.mean(), median_j=float(np.median(j)), mean_j=j.mean(), mean_j_net=n.mean(), mean_s=s.mean(),
            j_per_token=j.sum() / t.sum(), j_per_token_ci=boot(lambda r: [(lambda i: j[i].sum() / t[i].sum())(r.integers(0, m, m)) for _ in range(1000)]),
            token_ratio=t.mean() / eng["tokens"].mean(), energy_ratio=ratio, net_energy_ratio=n.mean() / eng["net"].mean(), latency_ratio=s.mean() / eng["seconds"].mean(),
            jpt_vs_english_pct=100 * (j.sum() / t.sum() / jpt_e - 1))
    T = np.array([r["tokens"] for r in rows]); J = np.array([r["joules"] for r in rows]); b, a = np.polyfit(T, J, 1)
    out["fit_all"] = dict(slope_j_per_token=b, intercept_j=a, r2=1 - ((J - (a + b * T)) ** 2).sum() / ((J - J.mean()) ** 2).sum())
    return out, rows

def figures(results):
    fig, ax = plt.subplots(figsize=(9, 4)); w = 0.8 / len(results)
    for i, (o, _) in enumerate(results):
        ls = [l for l in o["langs"] if l != "eng_Latn"]
        ax.bar(np.arange(len(ls)) + i * w, [o["langs"][l]["energy_ratio"] for l in ls], w, label=o["model"].split("/")[-1])
        ax.scatter(np.arange(len(ls)) + i * w, [o["langs"][l]["token_ratio"] for l in ls], marker="_", c="k", s=120, zorder=3)
    ax.axhline(1, c="gray", lw=.8); ax.set_xticks(np.arange(len(ls)) + .4 - w / 2); ax.set_xticklabels([l[:3] for l in ls])
    ax.set_ylabel("J/sentence vs English (bars); token ratio (black ticks)"); ax.legend(); fig.tight_layout(); fig.savefig("results/fig2_energy_ratio.png", dpi=150)
    fig, axs = plt.subplots(1, len(results), figsize=(5 * len(results), 4), squeeze=False)
    for ax, (o, rows) in zip(axs[0], results):
        for l in sorted({r["lang"] for r in rows}):
            ax.scatter([r["tokens"] for r in rows if r["lang"] == l], [r["joules"] for r in rows if r["lang"] == l], s=5, label=l[:3])
        f = o["fit_all"]; x = np.array([0, max(r["tokens"] for r in rows)]); ax.plot(x, f["intercept_j"] + f["slope_j_per_token"] * x, "k--")
        ax.set_title(f"{o['model'].split('/')[-1]}: {f['slope_j_per_token']:.2f} J/token, R²={f['r2']:.3f}"); ax.set_xlabel("tokens per sentence"); ax.set_ylabel("J per sentence")
    axs[0][0].legend(fontsize=6, ncol=2); fig.tight_layout(); fig.savefig("results/fig3_joules_vs_tokens.png", dpi=150)

if __name__ == "__main__":
    res = [analyse(p) for p in sorted(glob.glob(sys.argv[1]))]
    json.dump([o for o, _ in res], open(sys.argv[2], "w"), indent=1, default=float); figures(res)
    for o, _ in res:
        print(o["model"], "fit", {k: round(v, 3) for k, v in o["fit_all"].items()})
        for l, v in o["langs"].items(): print(f"  {l} tok={v['mean_tokens']:.1f} J={v['mean_j']:.1f} J/tok={v['j_per_token']:.3f} ({v['jpt_vs_english_pct']:+.1f}% vs eng) Eratio={v['energy_ratio']:.2f} Tratio={v['token_ratio']:.2f} Lratio={v['latency_ratio']:.2f}")
