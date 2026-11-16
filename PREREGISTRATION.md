# Pre-registration: Energy cost of the tokenizer tax in memory-bound decoding (Indian languages)

Written before any data was looked at or measured. Status of each item is marked **[verified]** or **[TO VERIFY]**.
Nothing below has been measured yet.

## Questions and hypotheses

- **H1 (token tax).** Per tokenizer, the median tokens-per-sentence ratio vs English exceeds 1.5x for most of the 10 Indian languages.
  *Falsified if* the best tokenizer is <1.5x for a majority (>=6/10) of languages. (GO/NO-GO 1; the story then becomes "which tokenizer choices still impose a tax".)
- **H2 (energy follows tokens).** In teacher-forced decoding at batch size 1, joules per sentence is linear in tokens per sentence, so joules per *token* is roughly constant across languages.
  *Falsified if* a per-model fit of J/sentence on tokens/sentence has R^2 < 0.9, or if the per-language median J/token differs from English by more than 15% (bootstrap CI excluding that band) for most languages.
- **H3 (energy ratio = token ratio).** The ratio of J/sentence to English is within 15% of the token-count ratio, per language and model.
  *Falsified if* the ratio of ratios lies outside [0.85, 1.15] for most languages. Latency is tested the same way.
- **H4 (model dependence).** The energy ratio vs English differs by model family because tokenizers differ.
  *Falsified if* the max-min spread of ratios across models is <10% of the mean for most languages.
- **H5 (speculative decoding).** Speculative decoding reduces J/sentence for English; acceptance rate is lower for Indian languages, so the Indian/English energy gap *widens*.
  *Falsified if* the gap shrinks or is unchanged (bootstrap CI of the change includes 0). Either outcome is reported as found.
  Exactness check: greedy output with and without speculation must be token-identical; mismatches are counted and reported, not hidden.

## Metrics

Tokens/sentence, chars/token, ratio to English; J/sentence, J/token, ms/sentence (raw and idle-subtracted); acceptance rate; R^2 and slope of J vs tokens. Medians over 3+ repeats with bootstrap 95% CI (10,000 resamples).
GO/NO-GO 2: repeats of the same configuration agree within ~5% (relative spread of the median).

## Data

FLORES+ devtest (`openlanguagedata/flores_plus`, gated, needs HF token + accepting terms) [TO VERIFY access; fallback `facebook/flores` FLORES-200].
Languages: eng_Latn, hin_Deva, ben_Beng, tam_Taml, tel_Telu, mar_Deva, guj_Gujr, kan_Knda, mal_Mlym, pan_Guru, ory_Orya.
**Sentence IDs** (fixed now, in `sentence_ids.json`): 200 indices into the 1012-sentence devtest, `sorted(random.Random(20261113).sample(range(1012), 200))`. The same indices are used for every language. Sentences are never selected by looking at the data.
If a sentence cannot be handled in some language, it is reported, not silently dropped.

## Models [all TO VERIFY: exact HF IDs, licences, fp16 fit on a 16 GB T4, transformers support]

Candidates (IDs come from secondary web sources only):
| Family | Target | Draft (same tokenizer) |
|---|---|---|
| Qwen3.5 | `Qwen/Qwen3.5-4B` | `Qwen/Qwen3.5-0.8B` |
| Gemma 4 | `google/gemma-4-E4B-it` or `E2B-it` | smaller E2B or Google's dedicated draft model |
| Llama 3.2 | `meta-llama/Llama-3.2-3B-Instruct` (gated) | `meta-llama/Llama-3.2-1B-Instruct` |

Known risks, to be checked before M2: Qwen3.5 may use a hybrid (linear-attention) architecture whose cache does not behave like a plain KV cache and may not support HF assisted generation; Gemma 4 "E" models carry large per-layer embedding tables so "effective" and total parameters differ. Reported draft size will be the counted `sum(p.numel())`, not the marketing name. If a family fails, it is replaced and the replacement is recorded in git history.

## Measurement protocol

Hardware: Kaggle T4, fp16 (recorded: GPU name, driver, clocks, temperature, versions).
Energy: `pynvml.nvmlDeviceGetTotalEnergyConsumption` (mJ), differenced around each run after `torch.cuda.synchronize()`. Fallback: >=20 Hz power sampling, integrated.
Idle power measured before each batch of runs; both raw and idle-subtracted reported.
Teacher-forced decode: fixed short prompt, then the reference sentence tokens fed one at a time with KV cache. Batch size 1 (8 if time allows). Warm-up, randomised language order, 3 repeats.
Prefill-only run measured separately to isolate prompt cost.
Note: NVML total energy is board-level and has ~10-100 ms update granularity on some parts, so per-sentence energy may be too coarse; if so, sentences are batched into timed blocks (decision recorded before scaling up).

## Mitigation (M4)

HF assisted generation, greedy, task: translate an English FLORES sentence into the target language, fixed prompt and max length. Alternatives if the gap widens: prompt-lookup/n-gram drafting, lowest-tax tokenizer per language.
`energy_gap_reduction_pct` is defined as `(gap_baseline - gap_spec) / gap_baseline * 100`, where gap = mean over Indian languages of (J/sentence ratio to English). Negative values mean the gap widened and are reported as such.

## What we will not do

No estimated numbers in place of measured ones; no dropping languages or models after seeing results; no changes to the hypotheses above (deviations go in a dated "Deviations" section appended below).

## Deviations
(none yet)

### 2026-11-14 — Llama 3.2 replaced by Sarvam-1
Meta's gated access for Llama 3.2 was not obtainable (the request form asks for an affiliation we did not provide). We did not use ungated third-party mirrors of the weights, to avoid sidestepping the gate. Replacement third family: `sarvamai/sarvam-1` (2.5B, Indic-focused tokenizer, ungated; confirmed 2,525,087,744 params via HF metadata). No same-tokenizer small sibling is known, so for M4 Sarvam-1 is tested with prompt-lookup/n-gram drafting only (not draft-model speculation). This was decided after seeing M1 token counts for Qwen3.5/Gemma 4 but before any energy data; Sarvam-1's M1 numbers are reported whatever they are.
Also: Gemma-4 target size (E2B vs E4B) still to be decided in M2 from T4 memory fit. M1 ratios use sum of tokens over the 200 sentences (per-sentence median ratio also stored in results/m1_token_counts.json).

### 2026-11-14 — Hardware changed: local RTX 3050 6GB Laptop GPU instead of Kaggle T4; Gemma 4 dropped
Decision by the user (project owner). The study runs locally on an NVIDIA RTX 3050 6GB Laptop GPU (fp16). Gemma 4 E2B (5.12B params, 9.5 GiB fp16) does not fit in 6 GiB, so it is dropped; no 8-bit/4-bit substitution (it would change the per-token memory traffic being measured). Families: Qwen3.5 (target `Qwen/Qwen3.5-2B`, 2.27B; draft `Qwen/Qwen3.5-0.8B`) and Sarvam-1 (2.53B; no draft model, prompt-lookup only in M4). M1 token counts for Gemma 4 are kept in the results as a tokenizer-only comparison. All energy results are for this one laptop GPU, whose clocks/power are not locked; the README must say it does not transfer to T4 numbers. Idle power, temperature and SM clock are logged per run.

### 2026-11-16 — Final deviations (supersedes the 2026-11-14 hardware entry)
All energy results were finally produced on a **Kaggle Tesla T4** (the original plan), not on the local 3050; the user then supplied a Kaggle notebook. Gemma 4 E2B fits on the T4 and is **back** in. Gemma 4 E2B, Qwen3.5-4B and Sarvam-1 are the three families. The local-3050 pilot files are kept as a development record only and are not used in any table. Other changes, decided after the pilots and before the full M3/M4 runs unless noted:
- **Sample sizes.** M3 is one pass over the pre-registered 200 sentences x 11 languages (not 3 repeats) to fit GPU time. Repeats were done only in the M2 reliability pilot (3 languages x 20 sentences x 3 repeats). M4 uses the first 20 of the 200 sentences per language, not all 200.
- **Qwen3.5 cannot do assisted generation in Hugging Face** (stateful model). The draft-model test therefore used Qwen3-4B + Qwen3-0.6B, and Gemma 4 E2B + Google's 78M MTP drafter. Qwen3.5's own M4 run produced degenerate 1-2 token outputs and is excluded (file kept, marked INVALID).
- **Stopping rule for M4**: generation stops at the first token containing a newline (token-level criterion), max 160 new tokens. The English row is translated from Hindi.
- **Added after seeing the M4 data (post hoc, labelled as a sensitivity analysis):** a second M4 analysis restricted to sentences whose baseline output ended naturally (3-159 new tokens), because many outputs hit the cap. The pre-registered analysis on all sentences is reported first.
- **H1 outcome:** by the rule written above, H1 is falsified, because Sarvam-1 (added after the Llama substitution) is under 1.5x for all 10 languages. For Qwen3.5 and Gemma 4 alone it holds.
- **Exactness check outcome:** speculative outputs were not always token-identical to the baseline (see README); mismatches are reported, not hidden.
- Draft-model parameter count is the counted `sum(p.numel())` of the loaded drafter: 77,731,328.
