> ### ⚠️ THESE NUMBERS ARE SIMULATED
>
> This file was produced by running `localmind compare` against
> [`tools/mock_ollama.py`](../tools/mock_ollama.py), a fake daemon that streams
> placeholder tokens at fixed rates. It exists to show you **the shape of the
> report** before you download a single model.
>
> The quality column is flat because every mock model returns the same canned
> text — that is expected here and would be a red flag with real models.
>
> Generate the real thing on your own hardware:
>
> ```bash
> localmind compare -m llama3.2:3b -m phi4-mini:3.8b -m mistral:7b
> ```

---

# Model Comparison Study

_Generated 2026-09-01T11:57:57_

## 1. Test machine

| Property | Value |
|---|---|
| OS | Linux 6.18.44-fc-v22 |
| CPU | x86_64 |
| Cores | 1 physical / 1 logical |
| RAM | 3.9 GB total, 3.61 GB free at start |
| GPU | none detected (CPU-only inference) |
| Python | 3.12.3 |

## 2. Protocol

- **Test set:** localmind-core-v1 v1.0 — 6 prompts across 1 categories (factual)
- **Repeats:** 1 per prompt per model
- **Decoding:** temperature 0.0, top_p 0.9, seed `42` — identical for every model
- **Isolation:** each model is unloaded after its run so memory readings do not overlap; each model gets its own discarded warm-up
- **Quality:** deterministic scorer (keyword coverage 45%, instruction compliance 40%, non-degeneracy 15%)

## 3. Headline results

| Model | Params | Quant | Disk MB | TTFT ms | Decode tok/s | Total ms | Peak RSS MB | Quality /100 |
|---|---|---|---|---|---|---|---|---|
| `mock-fast:1b` | 1.2B | Q4_K_M | 1049 | 80 | 47.58 | 879 | 22 | 54.6 |
| `mock-mid:3b` | 3.2B | Q4_K_M | 1907 | 172 | 21.89 | 1907 | 22 | 54.6 |
| `mock-mid:3b-q8` | 3.2B | Q8_0 | 3242 | 243 | 13.95 | 2966 | 22 | 54.6 |
| `mock-slow:7b` | 7.2B | Q4_K_M | 4196 | 454 | 7.49 | 5532 | 22 | 54.6 |

## 4. Rankings

| Criterion | Order (best first) |
|---|---|
| Fastest decode | `mock-fast:1b` > `mock-mid:3b` > `mock-mid:3b-q8` > `mock-slow:7b` |
| Lowest first-token latency | `mock-fast:1b` > `mock-mid:3b` > `mock-mid:3b-q8` > `mock-slow:7b` |
| Smallest memory footprint | `mock-fast:1b` > `mock-mid:3b` > `mock-mid:3b-q8` > `mock-slow:7b` |
| Highest quality score | `mock-fast:1b` > `mock-mid:3b` > `mock-mid:3b-q8` > `mock-slow:7b` |

**Quality per second of wait** (composite score ÷ median response time):

| Model | Quality/s | Quality | Median response (s) |
|---|---|---|---|
| `mock-fast:1b` | 62.14 | 54.6 | 0.88 |
| `mock-mid:3b` | 28.63 | 54.6 | 1.91 |
| `mock-mid:3b-q8` | 18.41 | 54.6 | 2.97 |
| `mock-slow:7b` | 9.87 | 54.6 | 5.53 |

## 5. Per-category quality

| Model | factual |
|---|---|
| `mock-fast:1b` | 54.6 |
| `mock-mid:3b` | 54.6 |
| `mock-mid:3b-q8` | 54.6 |
| `mock-slow:7b` | 54.6 |

## 6. Cold start

| Model | Weight load ms | TTFT ms | Total ms |
|---|---|---|---|
| `mock-fast:1b` | 52 | 81 | 397 |
| `mock-mid:3b` | 120 | 170 | 855 |
| `mock-mid:3b-q8` | 165 | 241 | 1315 |
| `mock-slow:7b` | 315 | 453 | 2459 |

## 7. Findings

- mock-fast:1b decodes at 47.6 tok/s versus 7.5 tok/s for mock-slow:7b — 6.35x faster.
- Quality is effectively tied at ~54.6/100 across all models (spread 0.0 points). When accuracy does not separate them, pick on speed and memory.
- Peak resident memory is within 5% across models (~22 MB). On a GPU box this is expected — check `ollama ps` for VRAM instead.
- mock-fast:1b wins on both speed and quality here, which is the easy case — take it unless memory is the constraint.

## 8. Selection guidance

- **Interactive chat / voice front-end** → `mock-fast:1b`: first-token latency dominates perceived quality here.
- **Batch extraction or offline document processing** → `mock-fast:1b`: nobody is watching the stream, so trade seconds for accuracy.
- **Running alongside an IDE, browser and containers on 16 GB** → `mock-fast:1b`: the memory headroom is worth more than the score.

- **Default pick if you only deploy one** → `mock-fast:1b`, which returns the most quality per second of user wait.

## 9. Threats to validity

- The quality scorer is heuristic. It rewards keyword coverage and instruction compliance, which correlates with usefulness but is not a substitute for human grading on open-ended prompts.
- On CPU-only machines RSS is the model footprint; on a discrete GPU most weights sit in VRAM and RSS understates the true cost — cross-check `ollama ps`.
- Laptop thermal throttling changes results between a cold and a hot chassis. Re-run back-to-back before drawing fine-grained conclusions.
- A single seed at temperature 0 removes sampling noise but does not remove prompt sensitivity. Widening the test set matters more than adding repeats.
