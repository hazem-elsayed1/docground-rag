# Retrieval configuration hash

The `config_sha256` field in retrieval comparison schema v2 hashes exactly the top-level JSON object named `config`. The complete object is stored in `report.json`. Metadata, timing, digests and the rest of the report are outside the hashed object. Arrays retain their original order.

Use Python's JSON encoder with lexicographically sorted object keys, literal Unicode, separators `(",", ":")` and nonfinite numbers disallowed. Encode the resulting string as UTF-8 without a BOM or trailing newline. Compute lowercase SHA-256 hex over those bytes.

~~~python
import hashlib
import json
from pathlib import Path

report = json.loads(Path("reports/retrieval-comparison-v2/report.json").read_bytes())
serialized = json.dumps(report["config"], sort_keys=True, ensure_ascii=False,
                        separators=(",", ":"), allow_nan=False)
assert hashlib.sha256(serialized.encode("utf-8")).hexdigest() == report["config_sha256"]
~~~

The v2 report also records these settings in `config_serialization`. Canonicalization here is defined by the Python encoder, including its number representation. Tooling that uses a different encoder should compare the serialized bytes before relying on its digest.

The v1 digest reproduces using `json.dumps(report["config"], sort_keys=True).encode("utf-8")` with Python's default whitespace and ASCII escaping. The historical report omitted that detail. It is preserved with its original hash. The v2 digest changed because of the documented serialization, while the visible configuration and retrieval outputs replayed exactly.
