# MiniLM document windows v1 results

**Decision: Keep baseline.** Applied the [predeclared rule](../../experiments/minilm-windows-v1.md) without changing it.

Specification commit `48752414dbfe1c229a967b81a7fa19b059f30844`. Benchmark implementation commit `636a57481f5f9b585a3d0abaf5d7bab8e8a5f731`.
The paired sample contains 20 answerable and four unanswerable development questions. The raw results include ranked chunk IDs, matched unit flags, per-question deltas and every timing sample.

## Retrieval at K=10

| Variant | Overlap precision | Overlap recall | MRR | Macro unit recall | Complete evidence |
|---|---:|---:|---:|---:|---:|
| Dense single | 0.1300 | 0.9167 | 0.7683 | 87.5% | 17/20 |
| Dense multi | 0.1250 | 0.8667 | 0.7071 | 82.5% | 16/20 |
| Hybrid single | 0.1450 | 0.9833 | 0.9000 | 95.0% | 19/20 |
| Hybrid multi | 0.1450 | 0.9833 | 0.7467 | 95.0% | 19/20 |

The overlap metrics count chunks intersecting annotated evidence. Unit recall and complete coverage require full containment according to the existing evaluator.

## Paired outcomes

### Dense

Complete coverage gained **0** questions and lost **1**. `leave-one-out-split-count`. Complete coverage was unchanged on the other 19 answerable questions. Ranked IDs were identical for 1/24 queries.

- Unit recall improved on no questions.
- Unit recall worsened on 1. `leave-one-out-split-count`.
- Unit recall unchanged on 19. `pipeline-nested-parameter-key`, `estimator-repr-text-mode`, `cross-validate-return-indices`, `confusion-matrix-axis-convention`, `top-k-accuracy-multilabel-support`, `random-state-none-global-rng`, `feature-union-duplicate-features`, `time-series-shuffle-autocorrelation`, `stratification-fold-score-spread`, `explained-variance-systematic-offset`, `log-loss-probability-input`, `pipeline-cache-reuse-and-inspection`, `cross-val-predict-partition-and-scoring`, `stratified-group-fold-goal-and-limit`, `r2-negative-baseline-and-comparability`, `mixed-text-categorical-fixed-validation`, `log-target-wrapper-and-growth-metric`, `leave-one-experiment-out-balanced-score`, `precision-recall-threshold-final-holdout`.

### Hybrid

Complete coverage was unchanged on all 20 answerable questions. Ranked IDs were identical for 0/24 queries.

- Unit recall improved on no questions.
- Unit recall worsened on no questions.
- Unit recall unchanged on 20. `pipeline-nested-parameter-key`, `estimator-repr-text-mode`, `leave-one-out-split-count`, `cross-validate-return-indices`, `confusion-matrix-axis-convention`, `top-k-accuracy-multilabel-support`, `random-state-none-global-rng`, `feature-union-duplicate-features`, `time-series-shuffle-autocorrelation`, `stratification-fold-score-spread`, `explained-variance-systematic-offset`, `log-loss-probability-input`, `pipeline-cache-reuse-and-inspection`, `cross-val-predict-partition-and-scoring`, `stratified-group-fold-goal-and-limit`, `r2-negative-baseline-and-comparability`, `mixed-text-categorical-fixed-validation`, `log-target-wrapper-and-growth-metric`, `leave-one-experiment-out-balanced-score`, `precision-recall-threshold-final-holdout`.

First relevant rank improved on two Dense questions and worsened on four. In Hybrid, it worsened on six and improved on none. On the Dense loss, the fully containing chunk went from rank 5 to outside the top ten. BM25 ranked that chunk first, and Hybrid retained it at rank 8 with windows, down from rank 2. This is one concrete case where the lexical component helped preserve complete evidence when Dense lost it.

### Previously truncated evidence

| Question and unit | Dense rank and coverage | Hybrid rank and coverage |
|---|---|---|
| `pipeline-cache-reuse-and-inspection` unit 2 | 1 (covered) → 2 (covered) | 1 (covered) → 2 (covered) |
| `cross-validate-return-indices` unit 1 | 2 (covered) → 2 (covered) | 1 (covered) → 1 (covered) |

Both single-window methods already cover the two audited units in their top ten. The windowed methods also cover them. This comparison gives no evidence that recovering those positions improved coverage. Changing a rank does not isolate the effect of recovering a particular token.

## Build and storage

| Strategy | Windows | Build seconds, three runs | Median seconds | NPZ bytes | Retained matrix bytes | Window vector bytes |
|---|---:|---|---:|---:|---:|---:|
| single | 228 | 30.176, 29.908, 27.473 | 29.908 | 353,836 | 350,208 | 350,208 |
| multi | 349 | 48.756, 31.662, 49.347 | 48.756 | 354,572 | 350,208 | 536,064 |

The trial build median was 1.630× baseline. Retained vector count and shape are identical. Window-vector bytes are measured buffer sizes. Native peak RSS was unmeasured.

## Warm query latency

| Variant | Samples | Median ms | P95 ms | Min–max ms | P95 range across passes, ms |
|---|---:|---:|---:|---:|---:|
| Dense single | 120 | 16.790 | 29.523 | 5.125–54.266 | 10.245–33.226 |
| Dense multi | 120 | 16.858 | 30.943 | 4.653–59.844 | 12.629–30.800 |
| Hybrid single | 120 | 18.728 | 36.175 | 5.277–101.552 | 12.610–49.375 |
| Hybrid multi | 120 | 18.821 | 35.371 | 6.333–55.870 | 13.206–44.704 |

The five warm passes include full query-time search. The build timings and the model startup are separate. The sequential timings are from one Windows host and do not characterize concurrent serving.

## Decision and reproduction

- Coverage gain: failed.
- No complete question loss: failed.
- Macro unit recall preserved: failed.
- Build cost: passed.
- Query p95 cost: passed.
- Artifact cost: passed.
- Baseline replay: passed.
- Within run rank stability: passed.
- Within run vector stability: passed.

These are paired observations on a small development set. An individual answerable question moves complete coverage by five percentage points. No statistical superiority or held-out improvement is established.

Run `python -m experiments.minilm_windows` from a clean checkout with the pinned corpus and local model prepared. It reads only the fixed development inputs and writes temporary vector artifacts. The JSON captures the environment, corpus, config, model and embedding digests. The original production retrieval paths and historical reports are kept as the reference. All three rebuilt baseline NPZ digests matched the existing production index.
