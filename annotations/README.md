# Human annotation guide

The first human-reviewed development dataset is [pilot-v1.json](development/pilot-v1.json). Its [review record](pilot-development-review.md) contains the expected answers, approved evidence, and corpus hashes. The four-document development benchmark uses the [approved schema-v2 dataset](development/multi-document-v1.json) and its [owner approval record](development-owner-approval.md). The frozen-test split is reserved for separately authorized evaluation. Its freeze record (kept in the original project history and not included in this publication copy) documents the version, digest, and local custody.

Prepare one draft per split against the pinned processed JSON. The helper uses the same text, offsets, and document revision as chunking. A different processed revision requires reviewing coordinates again. Changing chunk size leaves coordinates intact.

## Find evidence

From the repository root with the virtual environment active, list the sections, inspect a section by its displayed ID or section anchor, then look up an exact quotation.

```console
python -m src.annotations data/processed/common_pitfalls.json browse
python -m src.annotations data/processed/common_pitfalls.json browse --section SECTION_ID
python -m src.annotations data/processed/common_pitfalls.json find "exact text copied from the corpus" --section SECTION_ID
```

Omit `--section` to search the whole document. Search is case-sensitive and lists every valid occurrence in source order. Browse shows each text field's JSON Pointer, anchor, and global character range. The displayed text is a JSON string so escapes such as `\n` and `\r` make line endings explicit. Quotation lookup uses the actual characters. For selections that are cumbersome to pass to the shell, add the local character offsets to the start of the corresponding field. For a cross-field span, use each endpoint's own field start and pointer. Offsets count Unicode code points from zero. The start is included and the end is excluded. The characters at `start` and `end - 1` must belong to their declared fields in the same section. Two LF characters separate adjacent fields.

## Write and review a draft

1. Use [annotation schema v1](../README.md#annotation-schema-v1). Copy the revision printed by the helper into `corpus_revision`. Assign stable, unique IDs, use a consistent annotation version, and keep separate working drafts for `development` and `frozen-test`. Keep a review note outside the export with the intended answer or reason the question is unanswerable and any decisions about acceptable evidence.
2. For an **answerable** question, write a clear question in a reader's words. Check that the pinned chapter supports the entire intended answer. Select the shortest passages that still preserve necessary qualifications, context, or code. Copy the chosen coordinate object from `find` into `evidence_units` as an alternative for that unit.
3. For a **multi-evidence** question, use one required unit per independently necessary part of the answer. Every unit must be satisfied. Put equivalent passages in the same inner array only when each passage supports that unit on its own. Alternatives can come from different sections. If two passages are both needed, put them in separate units.
4. For an **unanswerable** question, use a plausible request whose full answer cannot be established from the pinned chapter. Check the chapter for support and record the missing fact or prerequisite in the review note. Set `answerable` to `false` and use `"evidence_units": []`. An unsuccessful literal search alone is insufficient.
5. Have a second person review answerability, verify each quoted excerpt and its context, and check for missing alternatives or near-duplicate questions across splits. Use the commands below to review the exact draft and export after agreement. The helper validates coordinates and schema. The human reviewer decides whether each passage actually supports the answer.

```console
python -m src.annotations data/processed/common_pitfalls.json review development.draft.json
python -m src.annotations data/processed/common_pitfalls.json export development.draft.json annotations/development/pilot-v1.json
```

`review` displays the question, answerability, required units, alternatives, coordinates, and exact excerpts. It does not write files. `export` validates again and creates a new UTF-8 JSON file. It refuses to overwrite any existing file. Keep reviewed development exports under `annotations/development/` and reviewed holdout exports under `annotations/frozen-test/`. For the holdout, use independent drafting and review, freeze the approved version before using it, and reserve it for the fixed evaluation protocol. Write exports to either location after human approval.
