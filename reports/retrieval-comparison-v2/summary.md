# Development retrieval comparison v2

The v2 replay reproduces every stored ranking, score, quality metric and structural result from [v1](../retrieval-comparison-v1/summary.md). The JSON report preserves all four cutoffs, category breakdowns and question-level diagnostics. Schema v2 makes the configuration digest independently reproducible. See the [hash specification](../../docs/retrieval-report-format.md).

| Method at K=10 | Overlap precision | Overlap recall | Overlap MRR | Macro Unit Recall | Complete evidence |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.1400 | 0.9583 | 0.8375 | 92.5% | 18/20 |
| Dense | 0.1300 | 0.9167 | 0.7683 | 87.5% | 17/20 |
| Hybrid | 0.1450 | 0.9833 | 0.9000 | 95.0% | 19/20 |
| Hybrid + reranker | 0.1300 | 0.9250 | 0.8292 | 87.5% | 16/20 |

Overlap metrics count retrieved chunks that intersect approved evidence spans. Macro Unit Recall weights each answerable question equally and requires full containment of an approved alternative for each counted unit. Complete Evidence Coverage requires all units. The latter two are the primary development results.

On this development sample, hybrid lost no complete-coverage question to BM25 at K=10 and recovered one additional fully covered question, `log-target-wrapper-and-growth-metric`. There are only 20 answerable questions and four unanswerable questions. This improvement is one question and the sample cannot establish general superiority. Both evidence metrics have a 95% structural ceiling because the only unit for `stratification-fold-score-spread` crosses a chunk boundary. The reranker remains optional.

## Exact-score ties and exploratory scores

RRF adds `1/(60 + rank)` for each candidate in each component list with up to 20 results. Ranks are one-based after deduplication within the list. A missing candidate contributes zero. The code sorts equal floating-point scores by ascending chunk ID before applying the cutoff. The ID derives from the document revision, section and coordinates. Replay analysis found 74 exact tie groups in the saved top-20 lists and 11 boundaries where a tie crossed an evaluated cutoff. Reversing ID order within ties changed no complete-coverage result.

Exploratory top-score AUCs on the 20 answerable and four unanswerable development items were 0.3625 for hybrid, 0.75 for BM25, 0.9375 for dense and 1.00 for reranking. These are descriptive results from four negatives. No threshold was selected. A score or result count alone is insufficient to establish answerability.

## Latency

The original first run recorded one reranking-stage wall time of 667.556 seconds for `stratified-group-fold-goal-and-limit`. The saved trace identifies the observation but does not establish its cause. The original first-run mean was 30.384 seconds and median was 2.683 seconds. Excluding that single unexplained outlier gives a first-run mean of 2.681 seconds. The independent historical second run had a median of 2.356 seconds. The v2 replay median was 6.660 seconds while other verification work was running. Retain all raw timings, including the outlier. The typical estimates above describe sequential local measurement. Concurrency performance and production capacity remain unmeasured.

## Tokenizer audit

[The tokenizer audit](embedding-truncation.json) used the exact pinned MiniLM WordPiece tokenizer and configured limit of 256 tokens including special tokens, excluding batch padding. Across 228 chunks, 115 were truncated and the largest untruncated input was 642 tokens. Two fully contained approved evidence alternatives had tokens omitted in their chunks. These were unit 2 of `pipeline-cache-reuse-and-inspection` and unit 1 of `cross-validate-return-indices`. This audit concerns encoder input. The stored chunks remain intact for sparse retrieval and generation.

Run `python -m scripts.compare_retrieval` and `python -m scripts.audit_embedding_truncation` from the repository root with prepared assets. Compare quality data separately from timing and provenance. At the time of this report, generation evaluation was pending.

**Later update:** Groq generation was subsequently evaluated on the development split. See the [Groq development results](../../README.md#generation-groq-evaluation). The retrieval findings above are unchanged.
