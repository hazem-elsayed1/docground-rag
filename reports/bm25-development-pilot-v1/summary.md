# First BM25 development benchmark

**Scope.** Human-reviewed development pilot `pilot-v1`, with eight answerable and two unanswerable questions from the pinned scikit-learn 1.9.1 Common Pitfalls chapter. The frozen-test split contained only its original empty placeholder. This run measured the published baseline.

## Quality and latency

| K | Precision@K | Recall@K | MRR@K | Complete evidence coverage | Run 1 mean ms | Run 2 mean ms | Pooled median ms | Pooled range ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.750 | 0.625 | 0.750 | 4/8 | 0.11528 | 0.12648 | 0.11830 | 0.0864 to 0.1590 |
| 3 | 0.375 | 0.875 | 0.875 | 6/8 | 0.13360 | 0.13554 | 0.13085 | 0.1095 to 0.1756 |
| 5 | 0.225 | 0.875 | 0.875 | 6/8 | 0.15380 | 0.16419 | 0.15750 | 0.1223 to 0.2074 |

All four quality metrics are macro-averages over the eight answerable questions. Precision divides each question's relevant hit count by the requested K. Recall divides that count by the number of unique relevant chunks for the question in the full artifact. MRR uses the inverse of the first relevant rank through K and gives zero if there is no hit. Complete coverage divides the number of questions with all required units covered by eight. Each unit requires a single retrieved chunk containing one full alternative span. Chunk relevance uses any positive overlap with annotated source characters in the same section. The per-question recall denominators appear below.

Each mean uses ten queries per run. The pooled median and range use 20 samples per K. The evaluator uses `perf_counter_ns` around the ranker call on an already-built index. Timing includes tokenization, scoring, sorting, and materializing the returned hits. Process startup, index construction, artifact loading, annotation validation, metric scoring, and JSON serialization are excluded. Each K ran in its own process per repetition, in question order, with no explicit warm-up.

## Per-question diagnostics

`A` and `B` identify the approved evidence units. Each K column shows covered units followed by missing units. First relevant rank uses the overlap rule and one-based ranks. Every approved unit has a containing chunk, and the evaluator agreed with direct coordinate checks.

| Question | First relevant rank | Relevant chunks in corpus | K=1 covered / missing | K=3 covered / missing | K=5 covered / missing | Cause of misses |
| --- | ---: | ---: | --- | --- | --- | --- |
| Q1 | 1 | 1 | A, B / none | A, B / none | A, B / none | No misses. |
| Q2 | 1 | 1 | A / none | A / none | A / none | No misses. |
| Q3 | 1 | 1 | A / none | A / none | A / none | No misses. |
| Q4 | 2 | 2 | none / A, B | B / A | B / A | Ranking. A at rank 10, B at rank 2. |
| Q5 | 2 | 1 | none / A | A / none | A / none | Ranking. A at rank 2. |
| Q6 | 1 | 2 | A / B | A / B | A / B | Ranking. B at rank 13. |
| Q7 | 1 | 1 | A / none | A / none | A / none | No misses. |
| Q8 | 1 | 2 | A / B | A, B / none | A, B / none | Ranking. B at rank 3. |

### Top-five ranked chunks

Each entry gives the full chunk ID and BM25 score, ordered from rank 1 through 5. Scores here are rounded to six decimal places. The JSON retains full precision.

**Q1**

1. `2dbb1c1442058ad738c25476a5820890f4138f64ad4f0efdb2adcd3a4d6b406a` (14.975527)
2. `058c1c7cf6a52cce6e2b8a5ae23021ce0cd47f086a288e4aecf3add21aa01422` (11.879079)
3. `7a2706dd88b39c92993b5a182b3d4addb478b1282f7d03c1be8ba04f626e87ce` (9.670987)
4. `8b124a43e899ae185b0ca0d57b80877be15aa28e5de8291820c3952971c9a5bd` (8.503294)
5. `d4640d8ee6dc7016597dde75a8ac699459ab3b095552087ae0ada9e5ea8c2e8a` (8.339648)

**Q2**

1. `d347c0baa1401b62d5528b95b58842763522ee6e39015ba3ab93a34b2d67e6bb` (12.505058)
2. `058c1c7cf6a52cce6e2b8a5ae23021ce0cd47f086a288e4aecf3add21aa01422` (12.303594)
3. `8b124a43e899ae185b0ca0d57b80877be15aa28e5de8291820c3952971c9a5bd` (12.021415)
4. `34e25ba9f0530de8e30ec9fdb94eff06aaee31d5bf1b062563b7ccddf1dc40f7` (9.502798)
5. `7a2706dd88b39c92993b5a182b3d4addb478b1282f7d03c1be8ba04f626e87ce` (8.736690)

**Q3**

1. `02c6d3994b57f16b3328231b0e25d284d814acbf2b83799b8eb459b7f45bbb94` (12.494384)
2. `d4640d8ee6dc7016597dde75a8ac699459ab3b095552087ae0ada9e5ea8c2e8a` (11.676933)
3. `1f869dff9d09ed5e0885b2435c6192269d9bf975fdb41299c2a780c1a1ae9f90` (11.392566)
4. `4c7f2a914947dd0d8228754f301c8d3c9f42ee8bb8038cd6b44596b03e6966bb` (8.557665)
5. `afb406e3ff64b9fd60558810f487269503bc9fe469300e8844408ef94442fcf3` (8.457122)

**Q4**

1. `1a8e6b435984ed93ea1cd367885165abc438fc51b73350f9fb24f2fdfd6ac544` (12.362819)
2. `10a9e9efe2cb89ac3164f6bf1648295b0e1dbd8400ef4de1cf3f061e6a265507` (11.043012)
3. `41990be19fea70fddb52469d51600f10ffc28451e33ee190b4c74680a0bc41f3` (10.763057)
4. `fa2dc98510508dff4d8cd6d8fa7dce98bfff6e78bae60a9bc35c2a377bf2445d` (9.833656)
5. `a1461e81fd6db4eee4b1fe2aa615db4410372409503f7083344cd27f244dec00` (9.410236)

**Q5**

1. `7a2706dd88b39c92993b5a182b3d4addb478b1282f7d03c1be8ba04f626e87ce` (21.192741)
2. `41990be19fea70fddb52469d51600f10ffc28451e33ee190b4c74680a0bc41f3` (15.031993)
3. `2dbb1c1442058ad738c25476a5820890f4138f64ad4f0efdb2adcd3a4d6b406a` (13.532354)
4. `8b124a43e899ae185b0ca0d57b80877be15aa28e5de8291820c3952971c9a5bd` (13.519886)
5. `10a9e9efe2cb89ac3164f6bf1648295b0e1dbd8400ef4de1cf3f061e6a265507` (12.710447)

**Q6**

1. `d4640d8ee6dc7016597dde75a8ac699459ab3b095552087ae0ada9e5ea8c2e8a` (15.989174)
2. `4c7f2a914947dd0d8228754f301c8d3c9f42ee8bb8038cd6b44596b03e6966bb` (15.220918)
3. `a1461e81fd6db4eee4b1fe2aa615db4410372409503f7083344cd27f244dec00` (11.104750)
4. `06d479c140bb8e1ac6bd1dbac4c3ce900119e34f062b633b24c9fad6c3440172` (10.591983)
5. `a8ec042b124bc11842b44bdc771dfc3ee8facc0aba1ebeacd73243ef233eee95` (10.143546)

**Q7**

1. `4c7f2a914947dd0d8228754f301c8d3c9f42ee8bb8038cd6b44596b03e6966bb` (38.263222)
2. `02c6d3994b57f16b3328231b0e25d284d814acbf2b83799b8eb459b7f45bbb94` (15.315566)
3. `d4640d8ee6dc7016597dde75a8ac699459ab3b095552087ae0ada9e5ea8c2e8a` (14.922623)
4. `41990be19fea70fddb52469d51600f10ffc28451e33ee190b4c74680a0bc41f3` (12.834655)
5. `1f869dff9d09ed5e0885b2435c6192269d9bf975fdb41299c2a780c1a1ae9f90` (12.232394)

**Q8**

1. `ed76c7e6dc70938f5d4b04676a63afaee497a3f07fd00c5729e1733fe8d1b81d` (18.089450)
2. `d4640d8ee6dc7016597dde75a8ac699459ab3b095552087ae0ada9e5ea8c2e8a` (14.636314)
3. `a1461e81fd6db4eee4b1fe2aa615db4410372409503f7083344cd27f244dec00` (14.476063)
4. `4c7f2a914947dd0d8228754f301c8d3c9f42ee8bb8038cd6b44596b03e6966bb` (14.321459)
5. `06d479c140bb8e1ac6bd1dbac4c3ce900119e34f062b633b24c9fad6c3440172` (12.348827)

### Unanswerable questions

| Question | K | Returned in runs 1 / 2 | Latency in run 1 (ms) | Latency in run 2 (ms) |
| --- | ---: | --- | ---: | ---: |
| Q9 | 1 | 1 / 1 | 0.0864 | 0.0967 |
| Q9 | 3 | 3 / 3 | 0.1109 | 0.1095 |
| Q9 | 5 | 5 / 5 | 0.1356 | 0.1375 |
| Q10 | 1 | 1 / 1 | 0.1050 | 0.1174 |
| Q10 | 3 | 3 / 3 | 0.1228 | 0.1301 |
| Q10 | 5 | 5 / 5 | 0.1538 | 0.1566 |

Both queries returned results at every K. These counts describe lexical retrieval and do not assess answerability or abstention.

## Experiment provenance and repeatability

Git commit `d1a7f4d25cba6dadc831115fa6a0b818c7476c45`, with a clean working tree before and after measurement. Python 3.14.6 on Windows (AMD64). BeautifulSoup 4.15.0, SoupSieve 4.1.2, typing-extensions 4.16.0, and pytest 9.2.0. BM25 and its evaluator use the Python standard library for retrieval and scoring.

Corpus revision `7c96bf1e360e8d3561b98313409323d2c074371cd71fa5599d50c247f03f652e`. Chunk schema v1, 27 chunks and a 1,200-character target. Annotation schema and evaluation protocol v1. BM25 used `k1=1.2`, `b=0.75` and `log1p((N - df + 0.5) / (df + 0.5))`. Tokenizer v1 uses Unicode word runs, casefolding, intact compound tokens and their underscore and ASCII case components, and deduplicated query terms. Ties use ascending chunk ID. The JSON records the full tokenizer configuration.

| Artifact | SHA-256 |
| --- | --- |
| `data/raw/common_pitfalls.html` | `21fac11c181ab35624dcf8673f0475d403d4f6ecf0f9bead01a573d7642b515c` |
| `data/processed/common_pitfalls.json` | `460654b1b7277ba589dd5856eecd1d871074889f4067f9d9766944ee6c8f27f5` |
| `data/processed/common_pitfalls.chunks.json` | `61bec3af779573331a224f30f0ed1857feb0f6a1893578fab837235973ac941f` |
| `annotations/development/pilot-v1.json` | `8b805eff016697513f8e2380db3ca52d7d1d3b58cb3ce13208bc79b1b9cfac16` |

Run from the repository root with the project virtual environment active. Repeat for K equal to 1, 3, and 5.

```console
python -m src.evaluation data/processed/common_pitfalls.chunks.json annotations/development/pilot-v1.json --top-k K
```

The two repetitions produced exactly equal full positive-score rank lists and BM25 scores, top-K IDs, per-question metrics, aggregate quality metrics, and coverage flags. Latencies varied. The [machine-readable report](report.json) contains the original evaluator output for both repetitions at all three cutoffs, plus the coordinate-based diagnostics and top-five scores.

## Interpretation and handoff

Ranks 2 and 3 add three relevant chunks in this pilot. Ranks 4 and 5 add none, so precision falls as K increases while recall, MRR, and full coverage plateau after K=3. Q4 and Q6 each have a relevant hit by rank 2, but still lack a required unit at K=5. Q4-A ranks 10th and Q6-B ranks 13th. No miss arose from a span crossing a chunk boundary or from a coordinate-matching discrepancy.

The result covers one technical chapter and eight answerable questions. Each question changes complete coverage by 12.5 percentage points. The development pilot supplies an initial measurement. Generalization and latency under deployment conditions remain unmeasured.

For independent Claude review, verify the input hashes, run the unmodified evaluator twice, and compare quality and ranked IDs while allowing latency to vary. Recheck Q4-A and Q6-B with the current chunk coordinates and BM25 ranks. Check that units require full containment even though the relevance metrics use overlap. Keep the frozen-test split out of this experiment.
