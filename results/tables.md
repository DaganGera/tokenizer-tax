### Table 1. Tokens vs English (M1, same 200 sentences)

| language | qwen3.5 | gemma4 | sarvam1 | cl100k_base | o200k_base |
|---|---|---|---|---|---|
| hin | 2.04 | 1.32 | 1.15 | 4.84 | 1.59 |
| ben | 2.21 | 1.22 | 1.32 | 5.93 | 1.72 |
| tam | 2.59 | 1.46 | 1.17 | 7.65 | 1.96 |
| tel | 2.88 | 1.80 | 1.20 | 8.37 | 1.95 |
| mar | 2.33 | 1.42 | 1.10 | 5.14 | 1.85 |
| guj | 3.11 | 1.88 | 1.23 | 7.75 | 1.81 |
| kan | 3.12 | 1.93 | 1.24 | 8.90 | 1.97 |
| mal | 3.12 | 1.89 | 1.39 | 9.22 | 1.99 |
| pan | 3.90 | 2.74 | 1.41 | 8.01 | 2.66 |
| ory | 3.96 | 3.59 | 1.50 | 12.60 | 5.07 |
| langs under 1.5x | 0/10 | 4/10 | 10/10 | 0/10 | 0/10 |

### Table 2. Energy per sentence, ratio to English (M3, T4, batch 1, raw joules; token ratio in brackets)

| language | gemma-4-E2B-it | Qwen3.5-4B | sarvam-1 |
|---|---|---|---|
| hin | 1.32x (1.32x) | 2.05x (2.04x) | 1.16x (1.15x) |
| ben | 1.21x (1.22x) | 2.20x (2.21x) | 1.31x (1.32x) |
| tam | 1.45x (1.46x) | 2.58x (2.59x) | 1.16x (1.17x) |
| tel | 1.82x (1.80x) | 2.85x (2.88x) | 1.19x (1.20x) |
| mar | 1.42x (1.42x) | 2.33x (2.33x) | 1.10x (1.10x) |
| guj | 1.85x (1.88x) | 3.08x (3.11x) | 1.23x (1.23x) |
| kan | 1.92x (1.93x) | 3.09x (3.12x) | 1.25x (1.24x) |
| mal | 1.86x (1.89x) | 3.13x (3.12x) | 1.39x (1.39x) |
| pan | 2.73x (2.74x) | 3.90x (3.90x) | 1.41x (1.41x) |
| ory | 3.59x (3.59x) | 3.92x (3.96x) | 1.51x (1.50x) |
| English J/sentence | 121.1 J | 100.7 J | 68.6 J |

### Table 3. Is joules per token constant? (M3)

| model | parameters | slope J/token | R² | J/token vs English, range over 10 languages | ms/token | bandwidth floor ms (assumption) |
|---|---|---|---|---|---|---|
| gemma-4-E2B-it | 5.10B | 4.466 | 0.9987 | -1.2% to +0.8% | 69.3 | 31.9 |
| Qwen3.5-4B | 4.21B | 3.624 | 0.9991 | -1.0% to +0.6% | 54.6 | 26.3 |
| sarvam-1 | 2.53B | 2.189 | 0.9928 | -1.1% to +0.7% | 33.3 | 15.8 |

### Table 4a. Speculative decoding, all 220 sentences (pre-registered analysis)

| model | method | sentences | acceptance | J/sentence change, Indian | J/sentence change, English | gap reduction % (95% CI) | outputs identical to baseline |
|---|---|---|---|---|---|---|---|
| gemma-4-E2B-it | draft | 220 | 0.25 | -45% | -28% | +23.9 (+13.4 to +31.6) | 206/220 |
| gemma-4-E2B-it | prompt_lookup | 220 | 0.43 | -60% | -9% | +55.8 (+46.9 to +62.2) | 202/220 |
| Qwen3-4B | draft | 220 | 0.33 | +66% | +273% | +55.5 (+40.1 to +65.1) | 181/220 |
| Qwen3-4B | prompt_lookup | 220 | 0.06 | -23% | -4% | +19.6 (+15.1 to +23.6) | 205/220 |
| sarvam-1 | prompt_lookup | 220 | 0.09 | -7% | -1% | +6.1 (-0.8 to +12.6) | 201/220 |

### Table 4b. Speculative decoding, sensitivity: only sentences whose baseline output ended naturally

| model | method | sentences | acceptance | J/sentence change, Indian | J/sentence change, English | gap reduction % (95% CI) | outputs identical to baseline |
|---|---|---|---|---|---|---|---|
| gemma-4-E2B-it | draft | 172 | 0.18 | -24% | -28% | -5.4 (-18.7 to +6.8) | 159/172 |
| gemma-4-E2B-it | prompt_lookup | 172 | 0.25 | -33% | -9% | +26.2 (+13.1 to +36.0) | 159/172 |
| Qwen3-4B | draft | 71 | 0.34 | +80% | +275% | +51.9 (+35.4 to +63.1) | 42/71 |
| Qwen3-4B | prompt_lookup | 71 | 0.06 | -17% | -4% | +13.4 (+6.3 to +19.6) | 62/71 |
| sarvam-1 | prompt_lookup | 216 | 0.05 | -2% | -1% | +0.9 (-4.6 to +5.7) | 197/216 |
