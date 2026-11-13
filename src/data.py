"""Load the pre-registered FLORES sentences (same 200 IDs for every language)."""
import json, pathlib
from huggingface_hub import hf_hub_download

LANGS = "eng_Latn hin_Deva ben_Beng tam_Taml tel_Telu mar_Deva guj_Gujr kan_Knda mal_Mlym pan_Guru ory_Orya".split()
IDS = json.load(open(pathlib.Path(__file__).parent.parent / "sentence_ids.json"))["ids"]

def load_sentences(lang):
    """Return the 200 pre-registered devtest sentences for `lang` (FLORES+ jsonl)."""
    p = hf_hub_download("openlanguagedata/flores_plus", f"devtest/{lang}.jsonl", repo_type="dataset")
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    rows.sort(key=lambda r: r["id"])           # id = position in devtest, same across languages
    assert len(rows) == 1012, (lang, len(rows))
    return [rows[i]["text"] for i in IDS]
