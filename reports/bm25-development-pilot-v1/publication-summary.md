# BM25 development smoke test

**On a 10-question development pilot over one chapter (27 chunks), BM25 (k1=1.2, b=0.75) fully covered the required evidence for 6 of 8 answerable questions at k=3 and k=5. This is a smoke test with wide uncertainty. It is not a ranking of retrieval methods and not a performance or abstention claim.**

| K | Precision@K | Recall@K | MRR@K | Complete question coverage | Complete unit coverage |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.750 | 0.625 | 0.750 | 4/8 | 7/12 |
| 3 | 0.375 | 0.875 | 0.875 | 6/8 | 10/12 |
| 5 | 0.225 | 0.875 | 0.875 | 6/8 | 10/12 |

A single answerable question moves complete question coverage by 12.5 percentage points. The quality metrics average over eight answerable questions. Precision uses requested K as the per-question denominator. Recall uses all chunks overlapping approved evidence for each question, and MRR uses the first such hit through K. Overlap-based recall and MRR can give credit when only some evidence is present. Complete question coverage requires every required unit to have a retrieved chunk containing one full approved span. The unit column counts individually covered units out of 12 and is derived from the original per-question flags.

## What remained incomplete

- **Q4.** The approved A span, which states the training-only fit rule and recommends chaining the selector with the estimator, is wholly contained at rank 10. The rank-2 chunk covers B and includes code for the pipeline and its use in cross-validation. The rank-10 placement is a lexical ranking miss under the approved annotation. There is also a separate evidence-sufficiency question. The expected fold-wise transformation combines the pipeline and CV passages, and any claim that the rank-2 material alone suffices requires semantic review.
- **Q6.** The approved B span, the general repeated-call rule for integer seeds, is wholly contained at rank 13. The rank-2 chunk already recommends an integer-seeded shuffled KFold for comparable folds. This passage may independently support the splitter claim, so incomplete alternative annotation may contribute to the measured miss. The published evaluation uses the approved spans as recorded.
- All 12 approved spans have containing chunks. No observed miss requires a span to be stitched across chunk boundaries.
- Both unanswerable queries returned K results at every cutoff. No abstention capability was evaluated.
- Mean retrieval times were approximately 0.12–0.16 ms across runs and cutoffs. These are illustrative in-memory timings on a built index, with no explicit warm-up. They exclude preprocessing, index construction, validation, serialization, and serving costs. One chapter and eight answerable questions support no generalization claim.

## Archived results and publication changes

The [original machine-readable report](report.json) and [original measurement summary](summary.md) are preserved byte-for-byte.

| Original file | SHA-256 |
| --- | --- |
| report.json | bf69f4c0d3d9ea92c8115f59901caaf4039d85fd85041fc7278964a9fec47570 |
| summary.md | fdc4ca0d8af874585047cba5399df0588964019d56de726dd6db00387a4c5e65 |

This publication summary adds the approved headline, unit totals derived from the archived coverage flags, qualifications about Q4 and Q6, and limits on abstention, latency, and generalization. The archive retains all original commands, configuration, artifact hashes, raw evaluator outputs, and timings. The publication replay matched both archived outputs at K=1, 3, and 5 when timing fields were excluded. It also reproduced the 40 archived top-five scores and all 12 containing-chunk ranks.

## Provenance and review handoff

The evaluated implementation was Git commit d1a7f4d25cba6dadc831115fa6a0b818c7476c45. The original reports record the complete SHA-256 values for the raw HTML, processed JSON, chunk JSON, and pilot annotations, plus the canonical corpus revision, Python and dependency versions, BM25 and tokenizer settings, and the protocol v1 invocation. The report's historical source URL is the HTML's canonical URL. Reproduction begins at the [reviewer-verified versioned endpoint](https://scikit-learn.github.io/1.9/common_pitfalls.html) and checks the recorded digest before saving the response.

For independent review, verify the archive digests, then follow the [reproduction instructions](../../README.md#reproduce-the-bm25-development-smoke-test). Compare the quality metrics and rankings while allowing latency to vary. Check Q4-A and Q6-B against the original annotation coordinates and assess the possible alternative passages separately from the recorded scores.
