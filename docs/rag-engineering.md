# DocGround engineering and operations

## Data flow

~~~mermaid
flowchart TD
  Manifest --> Fetch[Hash-checked HTML snapshots]
  Fetch --> Parse[Sections, fields and source coordinates]
  Parse --> Chunks[228 revision-bound chunks]
  Chunks --> Sparse[BM25]
  Chunks --> Encoder[Local MiniLM vectors]
  Encoder --> Dense[Exact cosine search]
  Sparse --> Fusion[RRF]
  Dense --> Fusion
  Fusion --> Optional[Optional cross-encoder]
  Fusion --> Context
  Optional --> Context
  Sparse --> Context
  Dense --> Context
  Context --> Gemini[Gemini structured claims]
  Gemini --> Validate[Resolve references against supplied fields]
  Validate --> UI[Claims and sources, abstention or service error]
~~~

The application imports `src.runtime` and `src.generation`. It has no annotation loader. The offline retrieval benchmark uses the development-only loader from Milestone 6A. Historical single-document scoring still uses the original BM25 entry point.

## Installation and deployment

Use a CPU host with Python 3.14, a virtual environment, local disk and HTTPS access for initial setup. Execute the five commands in the README from the checkout root. For BM25 alone, skip model preparation and choose BM25 in the UI. Full installation needs two model files of about 90 MB each, Python wheels and temporary download space. Allow several GB of disk and RAM for provisioning. Actual memory consumption depends on candidate depth, text length and concurrent users. Each ONNX session uses two intra-op threads. The Streamlit process caches one selected retriever at a time.

Linux CI uses Python 3.14 and installs `requirements-app.txt`, which pins the app's direct dependencies, including the Gemini HTTP transport. The Windows lock captures the measured Python 3.14.6 environment. Install it on Windows with `python -m pip install -r requirements-windows-py314.lock`. The comparison JSON records all installed distribution versions. Run `python -m pip check` and `python -m pytest tests -v` after installing the full app requirements. The base `requirements.txt` is sufficient for the historical ingestion-only reproduction.

The test suite runs offline. Before pytest imports any test module, `tests/conftest.py` removes provider credentials and disables `.env` loading. It also blocks non-loopback traffic through Python's socket connect and datagram send calls, its DNS lookups, and the Windows asyncio Proactor. Any attempt fails the run, including one a test catches. Native extensions that open their own sockets remain outside this guard. The module docstring lists the covered entry points and limits. The Streamlit app and `python -m scripts.evaluate_generation collect` load `.env` without overriding existing variables. The `references` and `score` actions never read it.

Start a local instance with `python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501`. Open `http://127.0.0.1:8501`. Probe `/_stcore/health` for process liveness. Submit a query to check corpus/model readiness separately. For private hosting, run the same command under a service account with read-only corpus and model mounts and a supervisor. Bind to the interface required by the host and place authentication and HTTPS in front of remote traffic. Keep credentials in the host's secret store. The review release has no public deployment. Restart after replacing artifacts to clear the resource cache.

## Provider and privacy

Copy `.env.example` to a local `.env` or supply environment variables. `GEMINI_API_KEY` authenticates the Developer API. `GEMINI_MODEL` defaults to `gemini-3.1-flash-lite`. Set `DOCGROUND_FREE_TIER_CONFIRMED=true` only after verifying that the key belongs to a free-tier project with billing disabled and that the chosen model is eligible. The switch records operator confirmation. Model quotas, region eligibility and terms can change. A Google AI Pro subscription alone does not establish API eligibility. Current official model and pricing references are [Google's model catalog](https://ai.google.dev/gemini-api/docs/models) and [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing).

The hosted provider receives the current question and selected public documentation excerpts. Avoid entering personal or confidential information in free-tier prompts. Requests use a 30-second HTTP timeout per attempt. Interactive calls permit at most two attempts. The SDK repeats a request after HTTP 429, 500, 502, 503 or 504, or after a connection failure or timeout, so the longest interactive wait is about one minute. Evaluation uses one SDK attempt per recorded request. Authentication, permission and other rejected requests are never repeated. Neither are local request errors or malformed success responses, which stop a collection run. Content blocks and empty answers in a well-formed response are final outcomes for that question. Provider log fields carry known reason codes, class names or `unknown`, never raw provider values. The `src/gemini.py` docstring tabulates each outcome's code, retry eligibility and effect on collection. No tools or external search are available to the model. The provider adapter handles authentication, quota, transport, truncation and empty responses. Token usage is recorded when returned. Log records include model and usage fields on successful requests. User-visible errors omit raw SDK exceptions, prompts and keys. For SDK usage and output schemas see [Google GenAI SDK](https://googleapis.github.io/python-genai/) and [structured output](https://ai.google.dev/gemini-api/docs/structured-output).

## Retrieval and reproducibility

The pinned local encoder maps up to 256 WordPiece tokens to 384 dimensions by attention-masked mean pooling and L2 normalization. Both documents and queries follow this path. Exact dot products cover all 228 chunks. The embedding artifact records corpus revision, chunk digest, model revision, both model asset digests, versions, shape, pooling and a digest of the vector bytes. Rebuild on any mismatch. Asset verification happens before ONNX inference. Model and index files are ignored by Git and rebuild locally.

The local model avoids API quota, network latency during retrieval and a remote embedding cache. The six-layer encoder is small enough for exact CPU search. Gemini embeddings would introduce a remote dependency. Larger general-purpose encoders would require another measured comparison. The pairwise MS MARCO model scores a question and chunk together with 512-token truncation in batches of eight. Both models were developed primarily for English. See the [encoder model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and [reranker model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2).

Hybrid takes up to 20 results per component. For each chunk ID, RRF sums `1/(60 + rank)` over lists that contain it, using the one-based rank after deduplicating that list. A missing component contributes zero. Equal floating-point scores sort by ascending chunk ID before taking the requested prefix. Chunk IDs are deterministic hashes of document revision, section and coordinates in the pinned corpus. The optional reranker scores only the first 20 fused candidates. BM25 remains independently available. Dense can return neighbors for an irrelevant query.

From a clean implementation checkout with prepared assets, run the following twice, choosing distinct output filenames.

~~~sh
python -m scripts.compare_retrieval > run-1.json
python -m scripts.compare_retrieval > run-2.json
~~~

Compare the JSON objects after removing `stage_latency_ms`. Within one commit and environment the remaining fields should match. Across commits compare configuration, inputs, candidates and results separately from provenance. The archived report identifies its implementation commit. Exact floating-point scores can vary across CPU kernels or wheels. Raw model revisions, hashes and full component rankings are in the JSON. Embedding preparation and model loading are excluded from query latency. Per-query stages are sequential. Dense service latency is query encoding plus vector search, hybrid adds BM25 and fusion, reranking adds its pair inference and sort. No generation is included.

## Retrieval protocol

K is 1, 3, 5 or 10. Relevance for precision, recall and reciprocal rank uses overlap with approved spans in the correct source revision and section. Precision divides by K, recall by the number of relevant chunks, reciprocal rank uses the first relevant hit. Unit recall counts a unit once if a single hit fully contains any approved alternative. Macro averaging weights each answerable question equally. Complete evidence requires all units on a question. The four unanswerable questions have separate retrieval counts. An empty or nonempty ranking alone is insufficient to determine answerability. Structural misses arise when no corpus chunk fully contains the unit, even at unlimited K.

## Generation and human evaluation

Each query selects up to ten chunks. Context selection retains complete chunks under 16,000 serialized UTF-8 bytes and removes duplicate section/text entries. This conservative byte budget applies to evidence. The output cap is 4,096 model tokens. The prompt and JSON evidence occupy separate system and user roles. Previous turns are for display only. Malformed responses, invalid citations and output truncation have distinct messages and evaluation codes. A truncated response is discarded before parsing and recording it does not stop collection as an outage. The app distinguishes local abstention when no excerpts were selected from a model judgment of insufficient or conflicting evidence. Any invalid citation rejects the whole response before display. No numerical answerability threshold has been calibrated.

The model returns a status and atomic claims with explicit/inference labels and citations. The application resolves each reference to a single supplied chunk, exact document revision and unique verbatim match within one source field. The server computes the coordinates and versioned URL. Ambiguous repeated quotes are rejected. Semantic support requires human review. The UI explicitly identifies generated claims as unreviewed and flags evidence removed by the context byte budget.

The development reference command validates all 24 reviewed IDs. Follow the [citation and outage protocol](citation-and-outage-protocol.md) to run the controlled development collection, complete claim-level judgments and score the results. The [Groq development results](../README.md#generation-groq-evaluation) report one scored run of `openai/gpt-oss-20b`, using AI-assisted judgments. Gemini, which the Streamlit app uses, remains unscored. Live prompt-injection resistance and contradiction handling remain unmeasured for both providers.

## Review and holdout protocol

Review code and development decisions locally, then obtain independent review. Resolve findings using development examples. Before requesting owner authorization, lock the exact code tree, input and configuration hashes, encoder and reranker revisions, prompt, provider model/version availability, method, context limits, scoring rubric and handling of service errors. Record the chosen method and rationale. Keep private labels, drafts and approvals outside the application, reports, CI and review bundle. After explicit owner approval only, use an isolated one-shot holdout runner on the locked system. Retain private per-question results locally and review aggregate publication separately. Any post-holdout design change requires a new validation plan.

## Limitations for adversarial review

The development set has 20 answerable questions. A single changed answer moves complete evidence coverage five percentage points. Development selection cannot estimate holdout performance. One approved span crosses a chunk boundary. The saved MiniLM audit found 115 of 228 chunks exceed the 256-token encoder limit, with approved evidence tokens omitted in two chunks. This concerns encoder input. The original chunk text remains available for BM25 and generation. See the [tokenizer audit](../reports/retrieval-comparison-v2/embedding-truncation.json). RRF discards absolute score gaps, and cross-encoder scores do not measure answerability. Synthetic provider tests establish protocol handling. Live quality, prompt-injection resistance, quota behavior and conflict detection need separate measurement. A valid quoted passage can be irrelevant to the generated claim. The configuration switch cannot inspect a Google project's billing state. Streamlit session history is in process memory and capped at five turns. Process health indicates local process liveness. Hosted deployment, concurrent serving and live Gemini quality require separate verification.
