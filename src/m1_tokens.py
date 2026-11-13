"""M1: tokens/sentence, ratio to English, chars/token per tokenizer x language. CPU only."""
import json, sys, statistics as st
from transformers import AutoTokenizer
from data import LANGS, load_sentences

# One tokenizer per model family (target and draft share it), plus tiktoken references from the paper.
HF_TOKENIZERS = {"qwen3.5": "Qwen/Qwen3.5-4B", "gemma4": "google/gemma-4-E2B-it",
                 "sarvam1": "sarvamai/sarvam-1"}
TIKTOKEN = ["cl100k_base", "o200k_base"]

def get_encoders():
    enc, skipped = {}, {}
    for name, repo in HF_TOKENIZERS.items():
        try:
            t = AutoTokenizer.from_pretrained(repo)
            enc[name] = lambda s, t=t: len(t.encode(s, add_special_tokens=False))
        except Exception as e:
            skipped[name] = f"{type(e).__name__}"
    import tiktoken
    for name in TIKTOKEN:
        t = tiktoken.get_encoding(name)
        enc[name] = lambda s, t=t: len(t.encode(s))
    return enc, skipped

def main():
    sents = {l: load_sentences(l) for l in LANGS}
    enc, skipped = get_encoders()
    out = {"skipped_tokenizers": skipped, "results": {}}
    for tok, f in enc.items():
        counts = {l: [f(s) for s in sents[l]] for l in LANGS}
        eng = counts["eng_Latn"]
        r = {}
        for l in LANGS:
            chars = sum(len(s) for s in sents[l])
            r[l] = {"mean_tokens": st.mean(counts[l]), "median_tokens": st.median(counts[l]),
                    "chars_per_token": chars / sum(counts[l]),
                    "ratio_to_english": sum(counts[l]) / sum(eng),
                    "per_sentence_ratio_median": st.median(a / b for a, b in zip(counts[l], eng))}
        out["results"][tok] = {"per_lang": r, "counts": counts}
    json.dump(out, open("results/m1_token_counts.json", "w"), ensure_ascii=False)
    print("skipped:", skipped)
    print(f"{'tokenizer':10}" + "".join(f"{l[:3]:>7}" for l in LANGS[1:]) + "   #langs<1.5x")
    for tok, v in out["results"].items():
        rs = [v["per_lang"][l]["ratio_to_english"] for l in LANGS[1:]]
        print(f"{tok:10}" + "".join(f"{x:7.2f}" for x in rs) + f"   {sum(x < 1.5 for x in rs)}/10")

if __name__ == "__main__":
    main()
