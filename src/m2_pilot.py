"""M2 pilot: J/sentence, latency, J/token for teacher-forced decoding. Randomised language order, repeats, bootstrap CI."""
import argparse, json, random, time, statistics as st, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import LANGS, load_sentences
from energy import Meter
from decode import PROMPT, prefill, teacher_forced_decode

def boot_ci(x, n=10000, seed=0):
    r = np.random.default_rng(seed); x = np.asarray(x)
    m = [np.median(r.choice(x, len(x))) for _ in range(n)]; return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--model", required=True); a.add_argument("--n", type=int, default=20)
    a.add_argument("--repeats", type=int, default=3); a.add_argument("--langs", default=",".join(LANGS))
    a.add_argument("--out", default="results/m2_pilot.json"); a = a.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).cuda().eval()
    meter = Meter(); langs = a.langs.split(",")
    ids = {l: [tok.encode(s, add_special_tokens=False) for s in load_sentences(l)[:a.n]] for l in langs}
    prompt = torch.tensor([tok.encode(PROMPT, add_special_tokens=False)], device="cuda")
    for l in langs[:2]:                                    # warm-up (not recorded)
        for r in ids[l][:3]: teacher_forced_decode(model, prefill(model, prompt), r)
    rows, rng = [], random.Random(0)
    for rep in range(a.repeats):
        idle = meter.idle_power(5)
        order = langs[:]; rng.shuffle(order)               # randomised language order each repeat
        for l in order:
            for i, r in enumerate(ids[l]):
                cache = prefill(model, prompt); torch.cuda.synchronize()
                with meter as m: teacher_forced_decode(model, cache, r)
                rows.append(dict(rep=rep, lang=l, sent=i, tokens=len(r), joules=m.joules, seconds=m.seconds,
                                 idle_w=idle, joules_net=m.joules - idle * m.seconds, **{k: v for k, v in meter.info().items() if k in ("sm_clock_mhz", "temp_c")}))
    summ = {}
    for l in langs:
        per_rep = [st.median([x["joules"] for x in rows if x["lang"] == l and x["rep"] == r]) for r in range(a.repeats)]
        allj = [x["joules"] for x in rows if x["lang"] == l]
        summ[l] = dict(median_j=st.median(allj), ci95=boot_ci(allj), rep_medians=per_rep,
                       rep_spread_pct=100 * (max(per_rep) - min(per_rep)) / st.median(per_rep),
                       mean_tokens=st.mean(x["tokens"] for x in rows if x["lang"] == l and x["rep"] == 0),
                       median_s=st.median(x["seconds"] for x in rows if x["lang"] == l))
    json.dump(dict(model=a.model, meter=meter.info(), n=a.n, summary=summ, rows=rows), open(a.out, "w"))
    for l, s in summ.items(): print(f"{l} tok={s['mean_tokens']:.1f} J={s['median_j']:.3f} CI={s['ci95'][0]:.3f}-{s['ci95'][1]:.3f} spread={s['rep_spread_pct']:.1f}%  s={s['median_s']:.3f}")

if __name__ == "__main__": main()
