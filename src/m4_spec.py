"""M4: real greedy translation with/without speculative decoding. Records J, latency, acceptance, and output identity."""
import argparse, json, random, statistics as st, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import candidate_generator as cg
from data import LANGS, load_sentences
from energy import Meter

NAME = dict(hin_Deva="Hindi", ben_Beng="Bengali", tam_Taml="Tamil", tel_Telu="Telugu", mar_Deva="Marathi", guj_Gujr="Gujarati",
            kan_Knda="Kannada", mal_Mlym="Malayalam", pan_Guru="Punjabi", ory_Orya="Odia", eng_Latn="English")
MAX_NEW = 160

def prompt_for(lang, src_text, src_name):   # fixed prompt; English target is translated from Hindi (no English->English task)
    return f"Translate the following {src_name} sentence into {NAME[lang]}. Output only the translation.\n{src_name}: {src_text}\n{NAME[lang]}:"

class Acc:                                  # records candidate tokens proposed / accepted by the assistant model
    proposed = accepted = 0
    @classmethod
    def patch(cls):
        for c in [v for v in vars(cg).values() if isinstance(v, type) and "update_candidate_strategy" in vars(v)]:
            print("acceptance counter patched on", c.__name__)
            orig = c.update_candidate_strategy
            def f(self, input_ids, scores, num_matches, _o=orig):
                cls.proposed += int(getattr(self, "num_assistant_tokens", getattr(self, "num_output_tokens", 0)))
                cls.accepted += int(num_matches); return _o(self, input_ids, scores, num_matches)
            c.update_candidate_strategy = f

def main():
    a = argparse.ArgumentParser(); a.add_argument("--model", required=True); a.add_argument("--draft", default=None)
    a.add_argument("--n", type=int, default=30); a.add_argument("--out", required=True); a = a.parse_args()
    Acc.patch(); tok = AutoTokenizer.from_pretrained(a.model)
    load = lambda r: AutoModelForCausalLM.from_pretrained(r, dtype=torch.float16).cuda().eval()
    model = load(a.model); draft = load(a.draft) if a.draft else None; meter = Meter()
    sent = {l: load_sentences(l)[:a.n] for l in LANGS}
    methods = {"baseline": {}, "prompt_lookup": dict(prompt_lookup_num_tokens=10)}
    if draft: methods["draft"] = dict(assistant_model=draft)
    gen = lambda ids, kw: model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False, stop_strings=["\n"], tokenizer=tok, pad_token_id=tok.eos_token_id, **kw)
    first = tok(prompt_for("hin_Deva", sent["eng_Latn"][0], "English"), return_tensors="pt").input_ids.cuda()
    for m in list(methods):                                                        # warm-up; drop methods this model does not support
        try: gen(first, methods[m])
        except Exception as e: print("UNSUPPORTED", m, type(e).__name__, str(e)[:160]); methods.pop(m)
    unsupported = [m for m in ("prompt_lookup", "draft") if m not in methods and (m != "draft" or draft)]
    rows, rng = [], random.Random(0); order = [(l, i) for l in LANGS for i in range(a.n)]; rng.shuffle(order)
    for l, i in order:
        src, sn = (sent["hin_Deva"][i], "Hindi") if l == "eng_Latn" else (sent["eng_Latn"][i], "English")
        ids = tok(prompt_for(l, src, sn), return_tensors="pt").input_ids.cuda(); outs = {}
        for m, kw in rng.sample(list(methods.items()), len(methods)):              # random method order per sentence
            Acc.proposed = Acc.accepted = 0
            with meter as mt: o = gen(ids, kw)
            outs[m] = o[0, ids.shape[1]:].tolist()
            rows.append(dict(lang=l, sent=i, method=m, joules=mt.joules, seconds=mt.seconds, new_tokens=len(outs[m]),
                             proposed=Acc.proposed, accepted=Acc.accepted))
        for r in rows[-len(methods):]: r["identical_to_baseline"] = outs[r["method"]] == outs["baseline"]
        if len(rows) % 300 == 0: json.dump(rows, open(a.out + ".partial", "w"))
    json.dump(dict(model=a.model, draft=a.draft, draft_params=sum(p.numel() for p in draft.parameters()) if draft else None, rows=rows, unsupported=unsupported), open(a.out, "w"))
    print("unsupported:", unsupported)
    for m in methods:
        r = [x for x in rows if x["method"] == m]
        print(m, "J/sent", round(st.mean(x["joules"] for x in r), 2), "identical", sum(x["identical_to_baseline"] for x in r), "/", len(r),
              "accept", round(sum(x["accepted"] for x in r) / max(1, sum(x["proposed"] for x in r)), 3))

if __name__ == "__main__": main()
