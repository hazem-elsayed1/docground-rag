# MiniLM document windows v1

Predeclared before the first experimental inference. The approved development split and its historical results are the only labels and comparison data.

## Hypothesis and controls

Overlapping windows may recover semantic signal when a relevant passage appears beyond MiniLM's document cutoff. Averaging can also dilute signal from an otherwise strong prefix. The independent variable is the document embedding strategy. Use the fixed 228 chunks in original ID order, approved development annotations, pinned MiniLM weights and tokenizer, original query encoder, masked mean and L2 pooling, CPU backend, and exact cosine ranking. Use the same software environment and model session for each variant. Evaluate at K=10 with the existing development metrics. Hybrid uses the existing BM25, per-component depth 20, RRF constant 60 and stable chunk-ID ties. The cross-encoder remains outside this experiment.

Both strategies use the exact chunk text and add no title or metadata prefix. Text already present in a chunk remains eligible. The tokenizer performs the original WordPiece segmentation once before overflow slicing, with no decoding and retokenizing of substrings. The trial uses 256 positions including the original CLS and SEP tokens, leaving 254 content positions. Adjacent full windows overlap by 32 content tokens, with step 222. The tokenizer supplies shorter final windows, attention masks and segment IDs. Pad inference groups of eight windows to the longest item, with the existing right padding. Apply the original pooling and normalization to each window. Average the normalized vectors with equal weight and L2-normalize the chunk result. Persist one 384-dimensional float32 vector per original chunk. Single-window comparison uses the existing builder and original one-window truncation.

## Measurements

Run three interleaved, sequential builds per strategy after one shared warm-up inference. Build wall time includes token preparation, inference, pooling and NPZ serialization. The report includes per-build timings, intermediate window-vector bytes, steady-state matrix bytes, NPZ size and output digests. The intermediate bytes measure the materialized vectors and exclude tokenizer state, model memory and native ONNX peak RSS.

Compare single-window Dense, multi-window Dense, single-window Hybrid and multi-window Hybrid. Warm each pipeline once on all 24 development queries, then measure five timed passes per pipeline. Rotate method order by query and pass. Warm-query latency includes the ordinary search path, including query inference and, for Hybrid, BM25 and fusion. Report pooled median and p95 plus each pass's distribution and raw per-query samples. Separate build costs from query latency. Single-window ordered hits and cutoff metrics must replay the archived v2 report. Repeat rankings and per-strategy vector digests must be stable locally.

## Decision rule

The primary outcome is Complete Evidence Coverage@10. Secondary outcomes are Macro Unit Recall@10 and overlap Precision, Recall and MRR@10. Recommend adoption with owner approval only when all the following hold.

- Hybrid completes at least one additional question, or Hybrid ties and Dense completes at least two additional questions.
- Neither method loses complete coverage on an individual answerable question previously fully covered. Macro Unit Recall does not decrease for either method.
- Median build time is at most twice the baseline. Trial warm-query p95 is at most 1.25 times its own method baseline for each method. Trial NPZ size is at most 1.01 times baseline.

Otherwise keep the baseline when measurement and replay checks hold. Classify the run as inconclusive if archived baseline replay fails, results are unstable, or a required observation is unavailable. Apply this one rule to one window configuration. This is a 20-answerable-question development sample. Hybrid has an archived 19/20 structural ceiling, so a dense-only benefit must justify the added complexity. Evidence diagnostics examine the two previously truncated evidence alternatives without treating a rank change as causal proof. No holdout, model selection or production change is authorized.
