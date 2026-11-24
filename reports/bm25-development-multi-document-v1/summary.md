# Multi-document BM25 development baseline

BM25 fully covered the approved evidence for **16 of 20 answerable questions at K=5** and **18 of 20 at K=10**. At K=10 it covered 26 of 28 required units. One remaining unit first appears at rank 11. The other crosses a chunk boundary and has no containing chunk in the fixed corpus.

## Retrieval results

| K | Precision@K | Recall@K | MRR@K | Complete evidence | Macro Unit Recall@K | Covered units | Ranking-missed units |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.7500 | 0.5917 | 0.7500 | 10/20 (50%) | 0.6000 | 15/28 | 12 |
| 3 | 0.3833 | 0.8333 | 0.8250 | 15/20 (75%) | 0.8250 | 22/28 | 5 |
| 5 | 0.2600 | 0.9250 | 0.8375 | 16/20 (80%) | 0.8750 | 24/28 | 3 |
| 10 | 0.1400 | 0.9583 | 0.8375 | 18/20 (90%) | 0.9250 | 26/28 | 1 |

Evaluation protocol v1 credits overlap with approved evidence for Precision, Recall, and MRR. Precision divides by the requested K. Complete evidence requires every required unit to have one approved alternative fully contained in a retrieved chunk from the declared document revision and section. Macro Unit Recall averages the fraction of covered units for each answerable question. Multiple alternatives can satisfy the same unit and earn one credit. The covered-unit column gives raw counts across all 28 units. All quality averages use the 20 answerable questions.

### Complete evidence by category

| Category | K=1 | K=3 | K=5 | K=10 |
| --- | ---: | ---: | ---: | ---: |
| Direct factual | 5/6 | 6/6 | 6/6 | 6/6 |
| Conceptual or explanatory | 4/6 | 5/6 | 5/6 | 5/6 |
| Multi-evidence within one document | 1/4 | 3/4 | 4/4 | 4/4 |
| Cross-document | 0/4 | 1/4 | 1/4 | 3/4 |

The four unanswerable questions each returned K results at every cutoff, with zero no-result queries. These counts describe retrieval behavior. The benchmark makes no abstention measurement. The sample is for development and the category counts are small.

## Structural ceiling and ranking misses

Before indexing, all 228 chunks were checked for full containment of each approved alternative. **27 units are structurally containable and one is structurally uncontainable.** The maximum macro Unit Recall is **95%**, and the maximum complete evidence coverage is **19/20 (95%)**. The count of structurally containable units is **27/28 (96.43%)**, which weights units equally across the dataset.

The uncontainable unit is unit 1 of `stratification-fold-score-spread`. Its sole approved Cross-validation span `[18073, 18502)` crosses consecutive chunks with ranges `[17126, 18246)` and `[18248, 18502)`. The overlap-based metrics can credit these chunks individually.

At K=10, the only ranking miss was unit 2 of `log-target-wrapper-and-growth-metric`. Its containing chunk was at rank 11. Every structurally containable unit had at least one positive-scoring containing chunk in the full BM25 ranking. The JSON report lists each cutoff's per-question ranked chunk IDs, missing units, category metrics, and best containing ranks. Unit numbers are one-based and follow the approved annotation order. [Machine-readable report](report.json)

## Provenance and repeatability

| Benchmark input | SHA-256 |
| --- | --- |
| `annotations/development/multi-document-v1.json` | `9f7131bfeedf80cd82cb0f956f97cd7b9a5b28871843060c446fed80ff3510a2` |
| `data/processed/corpus-v1/chunks.json` | `cee1cbdfc77457a77a6a1741bd72937f8095c511f0b4c4208616367e7dcfa882` |
| `corpus/scikit-learn-1.9.1-v1.json` | `805962c9e6fcd5b077fae77664de5cfa51955563ee09de97fa3bde2c6a8fd04c` |

- Corpus revision `01021d79036af9e2a1da8a58ab50e354c4cc3197bfa4eaa55c96e45351f59dca`, with four documents, 228 chunks, and the 1,200-character chunking target.
- Implementation commit `35ea2ab733c2d1873cdc85495d5a77d1e0add408` and Git tree `87ae2f1ff0497663286eb28c059f1ea49c1903ac`. These identify the committed source and scripts used for measurement. A separate commit archives the benchmark artifacts.
- Python 3.14.6 on Windows 11. BeautifulSoup 4.15.0, SoupSieve 4.1.2, typing-extensions 4.16.0, and pytest 9.2.0. The JSON report records all installed package versions in the measurement environment.
- BM25 used `k1=1.2`, `b=0.75`, and `log1p((N - df + 0.5) / (df + 0.5))` with tokenizer v1 and ascending chunk ID as the score tie-breaker.

Two separate Python processes with hash seeds 11 and 97 produced identical JSON after removing `retrieval_latency_ms`. The archived `report.json` is the seed-11 run. The prior reviewed baseline also matched after excluding timing and the updated Git provenance fields.

| K | Run 1 mean retrieval ms | Run 2 mean retrieval ms |
| ---: | ---: | ---: |
| 1 | 0.7364 | 0.6368 |
| 3 | 0.7531 | 0.8338 |
| 5 | 0.7849 | 0.6865 |
| 10 | 0.8614 | 0.7903 |

Mean latency includes all 24 queries. It covers tokenization, ranking, and result construction on an already-built index, with no explicit warm-up. It excludes corpus preparation, evidence analysis, scoring, and serialization.

## Reproduce

Use Python 3.14.6 and the pinned corpus artifact. To reconstruct the artifact from the verified raw snapshots, follow **Rebuild the pinned multi-document corpus** in the repository README. Run in PowerShell from the recorded implementation checkout with the virtual environment active.

~~~powershell
python -m pip install -r requirements.txt soupsieve==4.1.2 typing-extensions==4.16.0
python -m pytest tests -q
$env:PYTHONHASHSEED = '11'
python -m scripts.benchmark_bm25_development | Set-Content -Encoding utf8 bm25-run-1.json
$env:PYTHONHASHSEED = '97'
python -m scripts.benchmark_bm25_development | Set-Content -Encoding utf8 bm25-run-2.json
Remove-Item Env:PYTHONHASHSEED
python -c "import json; from pathlib import Path; reports = [json.loads(Path(p).read_text(encoding='utf-8-sig')) for p in ('bm25-run-1.json', 'bm25-run-2.json')]; [r.pop('retrieval_latency_ms') for r in reports]; assert reports[0] == reports[1]"
~~~

The full local test run passed with 114 tests. The suite covers macro Unit Recall, alternatives, document identity, a split evidence span, input rejection, repeatability, and the historical single-document evaluator contract.
