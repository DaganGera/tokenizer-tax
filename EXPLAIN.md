# EXPLAIN.md — the ideas behind this study, in plain English

Read this top to bottom once. Each section ends with a one-line "whiteboard version".

## 1. Tokenization and BPE

A language model does not read characters. It reads integers called **tokens**. A tokenizer cuts text into pieces from a fixed vocabulary (about 30k to 260k pieces) and maps each piece to an integer.

**Byte-pair encoding (BPE)** builds that vocabulary from a training corpus:
1. Start with single bytes (or characters) as the vocabulary.
2. Find the most frequent adjacent pair in the corpus, for example `t`+`h`.
3. Add the merged piece `th` to the vocabulary and replace the pair everywhere.
4. Repeat until the vocabulary has the size you want.

At encode time the tokenizer applies the learned merges in order. A word that was common in the training corpus becomes one or two tokens. A word that was rare gets cut into many small pieces, and at worst into single bytes.

**Why Indian languages pay a tax.** A Devanagari, Tamil or Odia character takes 3 bytes in UTF-8. If the corpus the tokenizer was trained on contained little of that script, the merges for it were never learned, so one character can cost 2 or 3 tokens. The same sentence (FLORES is parallel, so the *content* is identical) then needs several times more tokens than English. We measure this ratio directly in M1 (`results/m1_token_counts.json`).

Whiteboard version: *tokens are learned chunks; scripts the tokenizer did not see get cut into tiny pieces; the sentence has the same meaning but more tokens.*

## 2. Why decoding is memory-bound

Generating text happens one token at a time. To produce token *t+1* the model must run a forward pass that reads **all its weights** from GPU memory (DRAM) and does a small amount of arithmetic with them, because at batch size 1 each weight is used for only one token.

Two numbers decide the speed:
- **Memory time** = bytes read ÷ memory bandwidth.
- **Compute time** = floating-point operations ÷ peak FLOP/s.

Worked example (assumptions, not measurements; the T4 bandwidth is the published spec of about 320 GB/s): a 4.66B-parameter model in fp16 is about 9.3 GB. Reading it once takes about 29 ms. The arithmetic is about 2 × 4.66 billion = 9.3 GFLOP per token, which a T4 does in well under a millisecond at its fp16 peak. So memory time dominates by more than an order of magnitude. This is called **memory-bound**.

Consequence: **the cost of one decode step barely depends on which token it is.** Cost per sentence ≈ (cost per step) × (number of tokens). That is the hypothesis this project tests: if the tokenizer needs 3× more tokens for Tamil, Tamil should cost about 3× more time and energy.

Important caveat we measured, not assumed: if the software is slow in other ways (many tiny GPU kernel launches, Python overhead, an unoptimised layer type), the step cost is set by that overhead instead of by memory. Cost per step is still roughly constant per token, so the main conclusion survives, but the label "memory-bound" would be wrong. We report ms per token against the bandwidth floor for each model so the reader can judge.

Whiteboard version: *every new token re-reads all the weights; reading is the bottleneck; so time and energy scale with token count.*

## 3. The KV cache

Attention needs, for every earlier token, a **key** and a **value** vector at every layer. Without a cache, generating token *t+1* would recompute them for all *t* earlier tokens, which is quadratic work overall.

The **KV cache** stores the keys and values already computed. At each step the model computes keys and values only for the one new token, appends them, and attends over the stored ones. Cost per step stays about constant in the number of weights read, plus a small term that grows with context length.

Two details matter here:
- **Prefill vs decode.** Processing the prompt is one big parallel pass (compute-heavy). Generating is many small sequential passes (memory-heavy). We measure prefill separately so prompt cost is not mixed into decode cost.
- **Teacher forcing.** In M2/M3 we feed the *reference* sentence's tokens one at a time through the model with its cache. That is exactly the per-token work of real decoding, but it removes the model's quality from the picture: every language decodes exactly its own reference tokens, so differences cannot come from the model producing longer or shorter outputs.
- **Qwen3.5 is different.** It mixes attention layers with linear-attention (state-space style) layers that keep a fixed-size recurrent state instead of a growing KV cache. That makes it "stateful", and Hugging Face's assisted generation refuses to run on it, so speculative decoding could not be tested on Qwen3.5 (we report that as a finding).

Whiteboard version: *store old keys/values so each step only processes the new token.*

## 4. Measuring energy with NVML

NVML is NVIDIA's management library. `nvmlDeviceGetTotalEnergyConsumption` returns the board's cumulative energy in millijoules since the driver loaded. We read it before and after a piece of work, with `torch.cuda.synchronize()` on both sides, so the GPU has really finished. The difference is the energy.

Pitfalls we handle:
- **GPU work is asynchronous.** Without the synchronize call you would stop the clock before the work is done.
- **Idle power is not free.** The board draws power even when idle. We measure idle power before each language block and report both **raw** joules and **idle-subtracted** joules.
- **Scope.** The counter covers the whole GPU board, not the CPU, RAM or the rest of the machine. Our numbers are GPU energy only.
- **Clock and temperature.** GPUs change clock speed with temperature and power. We log SM clock and temperature with every measurement. We do not lock clocks, which we list as a limitation.
- **Noise.** We use randomised language order, warm-up runs, repeated runs and bootstrap confidence intervals. The gate we set was that repeats of the same configuration agree within about 5%.
- **Fallback.** If the counter were unsupported, the code would sample power at 50 Hz and integrate. On the T4 and on the laptop GPU the counter was supported.

Whiteboard version: *energy = counter(after) − counter(before), with the GPU synchronised and idle power accounted for.*

## 5. Speculative decoding and acceptance rate

Because decoding is memory-bound, checking several tokens at once is almost as cheap as producing one: the weights are read once either way. Speculative decoding exploits that:
1. A cheap **drafter** proposes *k* next tokens.
2. The big **target** model scores all *k* in one forward pass.
3. Under greedy decoding, we keep the drafted tokens up to the first one where the target would have chosen differently, and use the target's own choice there. We always get at least one correct token.

The output should equal what the target would have produced alone. The speed-up depends on the **acceptance rate**: the fraction of drafted tokens that the target agrees with. High acceptance means several tokens per target pass. Low acceptance means you paid for the drafter and still advanced about one token.

Two kinds of drafter appear in this study:
- A **draft model** (a small neural network sharing the target's tokenizer). For Gemma 4 this is Google's 78M-parameter multi-token-prediction drafter; for Qwen3 it is the 0.6B model.
- **Prompt lookup**: no network at all. It looks for the last few generated tokens earlier in the prompt and proposes the tokens that followed there. It helps when the output copies from the input.

Why it might not help Indian languages equally: drafters are trained mostly on English, so for Tamil or Odia they agree with the target less often, and the tokens are also cut into smaller pieces, which are harder to predict. The earlier paper on speculative decoding across languages saw lower acceptance outside English. Whether energy follows is what M4 measures.

**"Exact outputs" is not automatic.** In theory greedy speculation reproduces the target's output. In practice the target computes the *k* tokens in one batched pass in fp16, and that gives slightly different floating-point values than single-token passes. A near-tie between two tokens can then flip. Our smoke tests saw some such mismatches, so we count token-identical outputs and report the number instead of assuming it.

Whiteboard version: *a cheap guesser proposes, the big model verifies in one pass; the saving depends on how often the guesses are right.*

## 6. Quiz: ten questions a PhD would ask

Try to answer each aloud before looking at `README.md`.

1. Why use teacher-forced decoding instead of letting each model generate? What does it remove, and what does it *not* capture about real use?
2. At batch size 1, what limits decoding speed on a T4: compute or memory bandwidth? Show the arithmetic for a 4B-parameter fp16 model. What would change at batch size 32?
3. Energy per sentence ≈ (joules per token) × (tokens per sentence). Name two reasons joules per token might *not* be constant across languages, and say how you tested each.
4. NVML energy is board-level. What is missing from it, and could that bias the English-versus-Tamil comparison?
5. Your measured ms per token is above the bandwidth floor. What explains the gap, and does it threaten the hypothesis or only the "memory-bound" label?
6. Why do the Indian-language tokens cost more in the *same* model: is it the tokenizer, the script, or both? How would you separate them? (Think: bytes per character, merge coverage.)
7. What exactly does acceptance rate measure, and why can a lower acceptance rate *increase* energy rather than just failing to save it?
8. Greedy speculative decoding is "exact" in theory. Why did we still see non-identical outputs, and how large was the effect in the measured data?
9. The "gap" is J(Indic)/J(English). Give a case where the gap shrinks while Indian-language energy goes *up*. Which extra numbers do we report to prevent that misreading?
10. Which results transfer to another GPU, and which do not? What would you change to claim something about production serving (hint: batching, longer contexts, fixed clocks)?
