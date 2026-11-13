import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from data import LANGS
R = json.load(open("results/m1_token_counts.json"))["results"]
fig, ax = plt.subplots(figsize=(9, 4)); w = 0.8 / len(R)
for i, (t, v) in enumerate(R.items()):
    ax.bar(np.arange(10) + i * w, [v["per_lang"][l]["ratio_to_english"] for l in LANGS[1:]], w, label=t)
ax.axhline(1.5, ls="--", c="k", lw=1); ax.set_xticks(np.arange(10) + 0.4 - w / 2); ax.set_xticklabels([l[:3] for l in LANGS[1:]])
ax.set_ylabel("tokens vs English (x)"); ax.set_yscale("log"); ax.legend(); fig.tight_layout(); fig.savefig("results/fig1_token_tax.png", dpi=150)
