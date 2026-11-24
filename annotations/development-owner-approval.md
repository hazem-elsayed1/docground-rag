# Owner approval record for the development dataset

**Status** · Approved for the schema-v2 development export.

| Review field | Value |
| --- | --- |
| Draft | [24 development questions and evidence mappings](development-question-draft.md) |
| Draft SHA-256 (UTF-8 with LF line endings) | `46b0c870545f850828c731c7866ce91d75f774c40ca2a53db9cae84125b10799` |
| Corpus | [Four pinned scikit-learn 1.9.1 chapters](../corpus/scikit-learn-1.9.1-v1.json) |
| Corpus revision | `01021d79036af9e2a1da8a58ab50e354c4cc3197bfa4eaa55c96e45351f59dca` |
| Development scope | 20 answerable, 4 corpus-scoped unanswerable (D21–D24) |
| Evidence policy | 28 required units across 20 questions. D12 and D17 each have one extra alternative. D20 requires the Model Evaluation curve unit and one of the two test-isolation spans. |
| Independent review | The owner reported that the independent semantic review cleared D20 and D22. |
| Owner decision | “Approve and authorize export” in response to the Milestone 5E sign-off request for the 24 questions and evidence mappings, explicitly including D22 as corpus-scoped unanswerable |
| Export | [multi-document-v1.json](development/multi-document-v1.json), schema v2 with family IDs |
| Export SHA-256 | `9f7131bfeedf80cd82cb0f956f97cd7b9a5b28871843060c446fed80ff3510a2` |

## Owner sign-off

> I approve the 24 development questions and evidence mappings in the draft identified above, including D22 as unanswerable within the four pinned chapters. I authorize the schema-v2 development export.

The draft linked above is the exact revision presented to the owner. Its status text records the state at sign-off. This record captures the subsequent decision.

## Verification handoff

Two independent rebuilds from the approved draft produced byte-identical CLI exports. The saved dataset matches both. All 31 spans resolved to their exact excerpts in the declared document revisions. D12 and D17 each retain one additional alternative. D20 retains two required units and two test-isolation locations for its second unit. Schema validation rejected a test-isolation span reassigned to a different document. D21–D24 have empty evidence lists.
