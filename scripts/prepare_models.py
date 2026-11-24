"""Download verified local models and build the pinned embedding artifact."""
from src.dense import build_embeddings
from src.neural import MODELS, Transformer, model_bytes
from src.runtime import VECTORS, load_corpus


if __name__ == "__main__":
    corpus = load_corpus()
    for role in MODELS:
        for filename in ("tokenizer.json", "onnx/model.onnx"):
            model_bytes(role, filename, fetch=True)
    build_embeddings(corpus, Transformer("encoder"), VECTORS)
    print(f"Indexed {len(corpus.chunks)} chunks at {VECTORS}")
