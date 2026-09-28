# PARM-Passage-Anchored-Relation-aware-Multi-hop-Retrieval-for-Evidence-Grounded-Question-Answering

# Reproducibility and implementation details

This focused supplement exposes the entity/relation extraction implementation, its complete prompt and inference settings, and the answer post-processing used by the current PARM source. It is a subset of the implementation, not an end-to-end PARM release or evidence that the manuscript's reported scores have been reproduced.

| Material | Location |
| --- | --- |
| Complete extraction prompt, entity/triple schemas, normalization and retry loop | [extraction.py](extraction.py) |
| Model selection, provider requests, JSON parsing and backend retries | [model_client.py](model_client.py) |
| Extraction/retrieval/generation defaults and release status | [settings.json](settings.json) |
| PARM answer prompt, deterministic type-specific post-processing, PARM answer prompt and multi-reference EM/F1 | [postprocessing.py](postprocessing.py) |
| Deterministic 2Wiki selection and export of IDs from a frozen manifest | [question_ids.py](question_ids.py) |
| Dependencies for the published modules | [requirements.txt](requirements.txt) |

## Offline extraction model and settings

The current 2Wiki graph-builder defaults to the vLLM backend with the served model name `qwen2.5-7b-instruct-local`, intended for Qwen2.5-7B-Instruct. The served name is an alias, not a checkpoint revision. The standalone `TripleExtractor` constructor defaults to `gemini-2.5-flash`; the full builder explicitly passes the configured model. Set the backend and model together when using this supplement. Exact historical checkpoint revisions and per-condition run manifests are not included, so these defaults should not be presented as independent verification of every reported experiment.

Each extraction input consists of a document title (`**title**`) followed by the distinct passages sharing that title. The graph-builder uses four document workers. Extraction uses temperature **0.2**, a maximum of **768 output tokens**, and at most **five extractor-loop attempts including the first call**, with exponential backoff starting at **2 seconds**. The output-token limit is overridable through `PARM_EXTRACTION_MAX_OUTPUT_TOKENS`. Model adapters and provider SDKs may also retry, so five extractor attempts are not necessarily five total HTTP requests.

The full prompt is the `EXTRACTION_PROMPT` string in `extraction.py`. It asks for at most 12 entities, 15 triples and descriptions of at most 20 words; these are prompt instructions rather than hard validation limits. `Entity`, `Triple` and `ExtractionResult` specify the output schema. Gemini receives a JSON MIME type and response schema. The OpenAI-compatible adapters request JSON in the prompt and parse the returned text locally; they do not send a strict `response_format` schema.

## Use the published extraction component

Python 3.11 is suitable for these files. Run from the repository root:

```bash
python -m pip install -r requirements.txt
export PARM_BACKEND=vllm
export PARM_VLLM_MODEL=qwen2.5-7b-instruct-local
```

The vLLM adapter defaults to the local endpoint `http://127.0.0.1:8000/v1`. Configure a running model service before calling it. For Gemini, set `PARM_BACKEND=gemini`, `PARM_GEMINI_MODEL=gemini-2.5-flash`, and supply the API credential through the environment.

```python
from extraction import TripleExtractor
from model_client import configured_generation_model

extractor = TripleExtractor(model_id=configured_generation_model())
result = extractor.extract("**Document title**\nDocument text.")
print(result.model_dump() if result is not None else None)
```

`postprocessing.constrained_answer` accepts a graph adapter exposing `generate(prompt, schema, max_tokens)` and an already constructed evidence package. It retains the PARM prompt and deterministic yes/no and comparison normalization. It does not perform retrieval. The answer-generation configuration is temperature 0 and at most 80 output tokens. Metrics lowercase text, remove ASCII punctuation and English articles, normalize whitespace, and take the maximum EM/F1 over reference answers.

## Question identifiers and availability

The manuscript specifies the first 3,000 2Wiki validation questions in original-ID order, partitioned into B0/B1/B2, and the first 1,000 MuSiQue validation questions in original order. `select_2wiki` exposes the current deterministic selection implementation. A newly generated selection is not a substitute for the exact ID lists used in an earlier experiment.

Historical evaluation ID lists and archived manifests were not available in the source prepared for this release. They are **not included** here. To export an existing frozen experiment manifest without resampling:

```bash
python question_ids.py --manifest data/fixed/manifest.json --output question_ids.csv
```

The exporter writes only subset labels, question IDs, original source IDs and source indices. It does not copy credentials, machine paths, questions, answers or full experiment manifests. The complete retrieval pipeline, graph-loading pipeline, frozen graphs and checkpoints are outside this focused release.

A statement consistent with this release is: "The entity-and-relation extraction implementation and prompt, current inference configuration, answer post-processing rules, evaluation helpers and a question-ID export utility are publicly available in the repository. Historical experiment manifests and evaluation question-ID lists are not included in this release." Do not describe the exact historical ID lists or the complete PARM implementation as publicly released on the basis of this directory alone.
