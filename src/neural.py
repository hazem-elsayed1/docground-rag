"""Pinned local Transformer inference on CPU."""
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


TOKEN_LIMITS = {"encoder": 256, "reranker": 512}
MODELS = {
    "encoder": ("sentence-transformers/all-MiniLM-L6-v2",
                "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
                "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
                "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"),
    "reranker": ("cross-encoder/ms-marco-MiniLM-L6-v2",
                 "233902d25c440f23af6f7d6e94d2946bac0bee0a",
                 "5d3e70fd0c9ff14b9b5169a51e957b7a9c74897afd0a35ce4bd318150c1d4d4a",
                 "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"),
}


def model_bytes(role: str, filename: str, fetch: bool = False) -> bytes:
    repo, revision, weights_hash, tokenizer_hash = MODELS[role]
    path = Path("models") / role / Path(filename).name
    if fetch and not path.exists():
        with urlopen(f"https://huggingface.co/{repo}/resolve/{revision}/{filename}", timeout=120) as response:
            payload = response.read(100_000_001)
    else:
        payload = path.read_bytes()
    expected = tokenizer_hash if filename == "tokenizer.json" else weights_hash
    if sha256(payload).hexdigest() != expected:
        raise ValueError(f"Model asset checksum mismatch: {path}")
    if fetch and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary.write(payload)
        Path(temporary.name).replace(path)
    return payload


def model_tokenizer(role: str) -> Tokenizer:
    tokenizer = Tokenizer.from_str(model_bytes(role, "tokenizer.json").decode())
    tokenizer.enable_truncation(TOKEN_LIMITS[role])
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    return tokenizer


class Transformer:
    def __init__(self, role: str):
        self.tokenizer = model_tokenizer(role)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(model_bytes(role, "onnx/model.onnx"), options,
                                            providers=["CPUExecutionProvider"])

    def batches(self, texts: list, size: int = 8):
        for start in range(0, len(texts), size):
            encoded = self.tokenizer.encode_batch(texts[start:start + size])
            arrays = {name: np.asarray([getattr(row, attribute) for row in encoded], dtype=np.int64)
                      for name, attribute in (("input_ids", "ids"), ("attention_mask", "attention_mask"),
                                              ("token_type_ids", "type_ids"))}
            feed = {node.name: arrays[node.name] for node in self.session.get_inputs()}
            yield self.session.run(None, feed)[0], arrays["attention_mask"]
