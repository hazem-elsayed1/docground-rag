# Four-document evaluation dataset proposal

This is the historical Milestone 5C proposal. The [workflow notes](#workflow-notes-from-milestones-5e-and-5f) record how the published development set and provisional holdout followed it.

**Decision requested.** Approve a target of 72 new questions, with 24 for development and 48 reserved for one frozen evaluation. The larger holdout gives each category several independent cases. This is a coverage-oriented sample of the pinned scikit-learn 1.9.1 corpus. Its category mix cannot estimate the frequency of real user requests. [Corpus manifest](../corpus/scikit-learn-1.9.1-v1.json)

## Sampling and split

Each cell gives **development / frozen-test** counts. For answerable items, assign the primary document by the main claim in the intended answer before splitting. Count each item once. The local corpus has 27 Common Pitfalls, 25 Compose, 46 Cross-validation and 130 Model Evaluation chunks. Equal primary-document quotas give the shorter chapters meaningful representation.

| Mutually exclusive category | Common Pitfalls | Compose | Cross-validation | Model Evaluation | All documents |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct fact, one unit | 1 / 3 | 2 / 3 | 1 / 3 | 2 / 3 | 6 / 12 |
| Concept or explanation, one unit | 2 / 3 | 1 / 3 | 2 / 3 | 1 / 3 | 6 / 12 |
| Multiple units, one document sufficient | 1 / 2 | 1 / 2 | 1 / 2 | 1 / 2 | 4 / 8 |
| Multiple units, multiple documents necessary | 1 / 2 | 1 / 2 | 1 / 2 | 1 / 2 | 4 / 8 |
| Unanswerable across the corpus | — | — | — | — | 4 / 8 |
| **Total** | **5 / 10** | **5 / 10** | **5 / 10** | **5 / 10** | **24 / 48** |

For the 12 cross-document items, aim for at least four distinct document pairs. Require a coherent reader task and verify that no single document supplies acceptable evidence for every unit. Reclassify an item if a newly discovered alternative makes one document sufficient. For unanswerables, mix missing facts with partially supported requests. Record exactly what the corpus cannot establish.

**Duplicate screen.** A curator groups intents by the fact, mechanism or decision needed and its essential qualifications. Compare intended answers and evidence across all categories, even when wording, estimator names or source chapters differ. Keep one item per family across the entire dataset. Check especially for preprocessing and leakage in Common Pitfalls, Compose and Cross-validation, and for multi-metric scoring in Cross-validation and Model Evaluation. Sharing a topic alone need not imply duplication.

**Reproducible allocation.** After human review, lock the roster of distinct families, stable ASCII family IDs, categories and primary documents. Within each table cell (one pool for unanswerables), sort ascending by the hex SHA-256 of UTF-8 `docground-5c-split-v1|<family_id>`, breaking ties by family ID. Assign the first development quota to development and the remainder to frozen-test. Retain the roster and allocation rule with the curator. Resolve shortfalls before locking or seek approval for revised quotas. Never reroll a split after seeing results.

## Annotation and independent review

1. **Screen for pilot leakage.** The curator compares proposed intents with all ten published pilot items, including its unanswerables. Exclude paraphrases, reused examples testing the same claim, and variants built around the Q4 pipeline or Q6 splitter failures. Authors select passages by reading the corpus. Keep pilot rankings, chunk boundaries and system outputs out of drafting and adjudication. [Pilot review](pilot-development-review.md) · [Published caveats](../reports/bm25-development-pilot-v1/publication-summary.md)
2. **Draft from canonical text.** Human A records the intended answer and one unit per independently necessary claim. Select the shortest sufficient span that retains its conditions. For every unit, check related sections across all four documents for alternatives. Accept a passage only if it independently supports the same claim, scope and qualifications. If two passages must be combined, use distinct required units. Exclude items whose answer depends on an unreadable formula, plot or external API page.
3. **Verify independently.** Give Human B the question and corpus without A’s answer, answerability decision or spans. B records an answer or missing fact and locates support. For each proposed unanswerable, both humans log relevant sections checked in every document, including synonyms, code and neighboring context. A failed literal search is insufficient. Compare records, resolve disagreements against the exact text, and log the rationale and both sign-offs. If agreement fails, drop the item and independently review its replacement.
4. **Check coordinates and custody.** Use the existing corpus annotation CLI and schema v2. For every alternative verify the corpus revision, manifest ID, processed document ID, section, endpoint block paths, and half-open document-local Unicode offsets against the exact quotation. Unanswerables have zero units. After approval, the curator records corpus revision, annotation version and hashes of both exports. Keep holdout questions, answers and review notes away from retrieval development until the evaluation protocol is locked. A reproducible split does not itself protect the holdout. [Human annotation guide](README.md) · [Corpus CLI and schema v2](../README.md#multi-document-annotations-and-evaluation)

## Measurement proposal

Existing Precision@K, Recall@K and MRR@K count even one character of overlap with approved evidence in the correct source and section. A fragment can therefore earn relevance, and adding equivalent locations can increase the number of relevant chunks in the recall denominator. Complete Evidence Coverage@K requires all units and hides the difference between recovering some units and recovering none. [Evaluation protocol](../README.md#evaluation-protocol-v1)

For a future evaluation, report **Unit Recall@K** alongside the historical measures. For each answerable question, divide the number of covered required units by its total required units, then average those fractions equally across answerable questions. A unit counts once if any one top-K chunk fully contains one approved alternative in the matching document revision and section. Two of three units gives 2/3 even if one unit appears in several chunks. Exclude unanswerables from this denominator and report their count separately. With zero answerables the metric is undefined. A span fragmented across chunks or an unannotated equivalent passage can still receive zero credit. This measure cannot establish abstention or answer quality.

## Handoff and acceptance

Approve the sampling matrix, family rule and reviewer roles before authoring. Before a later freeze, require all 72 items to pass independent review, the quotas and family assignments to reproduce, all spans to validate, and every unanswerable to have a four-document search record. Inspect cross-document necessity and alternative completeness explicitly. If 15 distinct primary intents for a short chapter would force near-duplicates, document the shortfall and revisit the quota. Subgroup counts will still be small, and the holdout shares its source documents with development.

## Workflow notes from Milestones 5E and 5F

- The workflow retained the 24 development and 48 frozen-test targets and the five category quotas. The approved development primary counts were 2 Common Pitfalls, 6 Compose, 7 Cross-validation, and 5 Model Evaluation. The provisional frozen-test primary counts are 2, 11, 12, and 15 respectively. Counts are based on distinct question intents and available evidence.
- The development set was drafted, independently reviewed and approved by the owner before the holdout was authored. The two sets were drafted separately and screened for duplicate question families. The proposed hash split over a combined roster was therefore not applied. [Development approval and verification](development-owner-approval.md)
- The frozen-test questions and candidate mappings are in a local review draft pending a consolidated independent review. The frozen-test annotation directory still has its placeholder. The original Unit Recall@K proposal remains a measurement design.
