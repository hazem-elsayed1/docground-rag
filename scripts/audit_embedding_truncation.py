"""Audit exact dense-encoder truncation for the pinned development corpus."""
import json
from pathlib import Path

from scripts.benchmark_bm25_development import INPUT_SHA256, load_inputs, provenance
from src.neural import MODELS, TOKEN_LIMITS, model_tokenizer
from src.retrieval import BM25


def measure(chunks, questions, tokenizer):
    tokenizer.no_padding()
    full = type(tokenizer).from_str(tokenizer.to_str())
    full.no_truncation()
    rows = []
    for chunk in chunks:
        all_tokens = full.encode(chunk["text"])
        encoded = tokenizer.encode(chunk["text"])
        kept = [(start, end) for (start, end), special in
                zip(encoded.offsets, encoded.special_tokens_mask, strict=True) if not special]
        truncated = len(encoded.ids) < len(all_tokens.ids)
        boundary = chunk["start"] + max((end for _, end in kept), default=0)
        spans = []
        for question in questions:
            for unit_index, alternatives in enumerate(question["evidence_units"], 1):
                for alternative_index, span in enumerate(alternatives, 1):
                    if ((span["document_id"], span["section_id"]) != (chunk["document_id"], chunk["section_id"])
                            or span["start"] >= chunk["end"] or span["end"] <= chunk["start"]):
                        continue
                    start = max(span["start"], chunk["start"])
                    end = min(span["end"], chunk["end"])
                    # Count only dropped tokens overlapping the in-chunk portion of this span.
                    omitted = any(left + chunk["start"] < end and start < right + chunk["start"]
                                  for (left, right), special in zip(all_tokens.offsets, all_tokens.special_tokens_mask)
                                  if not special and left + chunk["start"] >= boundary)
                    spans.append({"question_id": question["id"], "unit": unit_index, "alternative": alternative_index,
                                  "span_start": span["start"], "span_end": span["end"],
                                  "fully_contained": start == span["start"] and end == span["end"],
                                  "evidence_tokens_omitted": truncated and omitted})
        rows.append({"chunk_id": chunk["id"], "document_id": chunk["document_id"], "token_count": len(all_tokens.ids),
                     "retained_token_count": len(encoded.ids), "truncated": truncated,
                     "retained_text_end": boundary, "development_span_overlaps": spans})
    return rows


def run(root):
    annotations, _, payload = load_inputs(root)
    corpus = BM25.from_bytes(payload)
    rows = measure(corpus.chunks, annotations["questions"], model_tokenizer("encoder"))
    return provenance(root) | {
        "input_sha256": {p.as_posix(): h for p, h in INPUT_SHA256.items()},
        "corpus_revision": corpus.corpus_revision, "encoder": MODELS["encoder"][0],
        "encoder_revision": MODELS["encoder"][1], "tokenizer_sha256": MODELS["encoder"][3],
        "max_tokens": TOKEN_LIMITS["encoder"], "includes_special_tokens": True, "padding_counted": False,
        "chunks_truncated": sum(row["truncated"] for row in rows), "chunks": rows,
    }


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd().resolve()), indent=2, allow_nan=False))
