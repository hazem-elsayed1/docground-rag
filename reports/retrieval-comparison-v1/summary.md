# Retrieval comparison on the approved development split

All methods use the same 24 questions and pinned 228-chunk corpus. The five quality metrics average over 20 answerable questions. The JSON includes candidate scores, per-question rankings, category breakdowns, unanswerable retrieval counts and missed evidence units.

Benchmark commit: `67eabff40b07482c6f93b0be0e525717ea5e74b6`

Configuration SHA-256: `f9c1a3d942401afbdd50b442359256835584b0ec73b1e56ec5fbc99043e750f5`

Embedding artifact SHA-256: `274e533880d15224e141239899c705f03c450842f68c5cd8eae8d4c52a3990e8`

## Retrieval quality

| Method | K | Precision | Recall | MRR | Macro unit recall | Complete evidence |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 1 | 0.7500 | 0.5917 | 0.7500 | 60.0% | 10/20 |
| bm25 | 3 | 0.3833 | 0.8333 | 0.8250 | 82.5% | 15/20 |
| bm25 | 5 | 0.2600 | 0.9250 | 0.8375 | 87.5% | 16/20 |
| bm25 | 10 | 0.1400 | 0.9583 | 0.8375 | 92.5% | 18/20 |
| dense | 1 | 0.6500 | 0.5167 | 0.6500 | 52.5% | 9/20 |
| dense | 3 | 0.3500 | 0.7667 | 0.7583 | 75.0% | 13/20 |
| dense | 5 | 0.2600 | 0.9167 | 0.7683 | 87.5% | 17/20 |
| dense | 10 | 0.1300 | 0.9167 | 0.7683 | 87.5% | 17/20 |
| hybrid | 1 | 0.8000 | 0.6417 | 0.8000 | 65.0% | 11/20 |
| hybrid | 3 | 0.4333 | 0.9250 | 0.9000 | 87.5% | 16/20 |
| hybrid | 5 | 0.2800 | 0.9583 | 0.9000 | 92.5% | 18/20 |
| hybrid | 10 | 0.1450 | 0.9833 | 0.9000 | 95.0% | 19/20 |
| reranked | 1 | 0.7000 | 0.5750 | 0.7000 | 57.5% | 10/20 |
| reranked | 3 | 0.3667 | 0.8417 | 0.8167 | 80.0% | 14/20 |
| reranked | 5 | 0.2600 | 0.9250 | 0.8292 | 87.5% | 16/20 |
| reranked | 10 | 0.1300 | 0.9250 | 0.8292 | 87.5% | 16/20 |

Hybrid is the application default. At K=10 it recovers an additional unit for `log-target-wrapper-and-growth-metric` versus BM25. Its only missing unit is the sole required unit for `stratification-fold-score-spread`, whose span crosses a chunk boundary. The 19 other answerable questions contribute 1 each and that question contributes 0, giving (19 + 0) / 20 = 95% macro unit recall and 19 / 20 = 95% complete coverage. Across questions, 27 of 28 units are structurally containable.

The optional reranker reordered the same hybrid candidates. At K=10 it lost partial or complete unit coverage relative to BM25 on three questions, and gained one unit on one question. Relative to hybrid, it moved evidence outside the first ten on `mixed-text-categorical-fixed-validation`, `leave-one-experiment-out-balanced-score` and `precision-recall-threshold-final-holdout`. These development outcomes and the local CPU cost support keeping the stage optional.

## Query-stage latency

These measurements come from the second run on Windows CPU. They exclude model loading, corpus embedding, and generation. The package includes both raw repetitions. Run 1 includes an approximately 668-second wall-time outlier in one pair-scoring stage, so it is unsuitable for typical latency estimates.

| Stage | Mean ms | Median ms |
|---|---:|---:|
| bm25 | 1.69 | 1.44 |
| query embedding | 11.97 | 11.26 |
| cosine search | 0.30 | 0.26 |
| fusion | 0.10 | 0.08 |
| pair inference and sort | 2461.84 | 2355.72 |

## Reproduction

Prepare the pinned corpus and weights using the [engineering guide](../../docs/rag-engineering.md). Run the following from the repository root.

~~~sh
python -m scripts.compare_retrieval > run-1.json
python -m scripts.compare_retrieval > run-2.json
~~~

The two archived runs are identical after removing `stage_latency_ms`. BM25 scores and ranked chunk IDs match Milestone 6A at all four cutoffs.

A replay at `d4702150c5ec31da4deabb7082e6117644c02c47` matched model configuration, input and embedding digests, candidate scores and rankings, structural ceiling and every retrieval quality result.
