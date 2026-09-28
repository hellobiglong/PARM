import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

try:
    from google import genai
except Exception:
    genai = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


@dataclass
class _UsageMetadata:
    prompt_token_count: int = 0
    candidates_token_count: int = 0
    total_token_count: int = 0


@dataclass
class _Embedding:
    values: list[float]


@dataclass
class _EmbeddingResponse:
    embeddings: list[_Embedding]


@dataclass
class _LocalResponse:
    text: str = ""
    parsed: Any = None
    usage_metadata: _UsageMetadata = field(default_factory=_UsageMetadata)


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "local"}


def _extract_json_blob(text: str) -> str:
    text = text.strip()
    if text.startswith("{") or text.startswith("["):
        return text
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        return match.group(0)
    match = re.search(r"\[.*\]", text, flags=re.S)
    if match:
        return match.group(0)
    return text


class _LocalLLM:
    def __init__(self) -> None:
        self.model_id = os.getenv("QASA_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        self.max_new_tokens = int(os.getenv("QASA_LLM_MAX_NEW_TOKENS", "512"))
        self.device_map = os.getenv("QASA_LLM_DEVICE_MAP", "auto")
        self.dtype = os.getenv("QASA_LLM_DTYPE", "bfloat16")
        self._lock = threading.Lock()
        self._tokenizer = None
        self._model = None
        self._embed_tokenizer = None
        self._embed_model = None

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype.lower(), torch.bfloat16)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=self.device_map,
        )
        self._model.eval()

    def _prompt_text(self, contents: str, schema: Any | None = None) -> str:
        messages = []
        if schema is not None:
            messages.append({
                "role": "system",
                "content": (
                    "Return only valid JSON that matches the requested schema. "
                    "Do not include markdown, explanations, or code fences."
                ),
            })
        messages.append({"role": "user", "content": contents})
        if hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return contents

    def generate_content(self, model: str, contents: str, config: dict | None = None):
        config = config or {}
        with self._lock:
            self._load()
            prompt_text = self._prompt_text(contents, config.get("response_schema"))
            inputs = self._tokenizer(prompt_text, return_tensors="pt")
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            prompt_len = int(inputs["input_ids"].shape[-1])
            temperature = float(config.get("temperature", 0.0) or 0.0)
            max_new_tokens = int(config.get("max_output_tokens", self.max_new_tokens))
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0.0,
                "pad_token_id": self._tokenizer.eos_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
            }
            if temperature > 0.0:
                generation_kwargs["temperature"] = max(temperature, 1e-4)
                generation_kwargs["top_p"] = 0.9

            import torch
            with torch.inference_mode():
                output_ids = self._model.generate(**inputs, **generation_kwargs)

            generated_ids = output_ids[0][prompt_len:]
            text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            parsed = None
            schema = config.get("response_schema")
            if schema is not None:
                blob = _extract_json_blob(text)
                try:
                    if hasattr(schema, "model_validate_json"):
                        parsed = schema.model_validate_json(blob)
                    else:
                        parsed = schema.parse_raw(blob)
                except Exception:
                    try:
                        data = json.loads(blob)
                        fields = getattr(schema, "model_fields", {})
                        if isinstance(data, list) and "entities" in fields:
                            parsed = schema.model_validate({"entities": data})
                        elif isinstance(data, list) and "triples" in fields:
                            parsed = schema.model_validate({"triples": data})
                    except Exception:
                        parsed = None

            usage = _UsageMetadata(
                prompt_token_count=prompt_len,
                candidates_token_count=int(generated_ids.shape[-1]),
                total_token_count=prompt_len + int(generated_ids.shape[-1]),
            )
            return _LocalResponse(text=text, parsed=parsed, usage_metadata=usage)

    def embed_content(self, model: str, contents: str | list[str]):
        with self._lock:
            self._load_embedding_model()
            texts = [contents] if isinstance(contents, str) else list(contents)
            vectors = [self._embed_text(text) for text in texts]
            return _EmbeddingResponse(embeddings=[_Embedding(values=v) for v in vectors])

    def _load_embedding_model(self) -> None:
        if self._embed_model is not None and self._embed_tokenizer is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._embed_model_id = os.getenv("QASA_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        self._embed_device = os.getenv("QASA_EMBEDDING_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self._embed_tokenizer = AutoTokenizer.from_pretrained(self._embed_model_id, trust_remote_code=True)
        self._embed_model = AutoModel.from_pretrained(
            self._embed_model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self._embed_device.startswith("cuda") else torch.float32,
        )
        self._embed_model.to(self._embed_device)
        self._embed_model.eval()

    def _embed_text(self, text: str) -> list[float]:
        import torch
        import torch.nn.functional as F

        inputs = self._embed_tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=int(os.getenv("QASA_EMBEDDING_MAX_LENGTH", "512")),
            return_tensors="pt",
        )
        inputs = {k: v.to(self._embed_device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self._embed_model(**inputs)
            token_embeddings = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            pooled = torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = F.normalize(pooled, p=2, dim=1)
        return pooled[0].float().cpu().tolist()


class _LocalModels:
    def __init__(self) -> None:
        self._llm = _LocalLLM()

    def generate_content(self, model: str, contents: str, config: dict | None = None):
        return self._llm.generate_content(model=model, contents=contents, config=config)

    def embed_content(self, model: str, contents: str | list[str]):
        return self._llm.embed_content(model=model, contents=contents)


class _LocalClient:
    def __init__(self) -> None:
        self.models = _LocalModels()


class _VLLMModels:
    def __init__(self) -> None:
        self.base_url = os.getenv("QASA_VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
        self.api_key = os.getenv("QASA_VLLM_API_KEY", "local")
        self.model = os.getenv(
            "QASA_VLLM_MODEL",
            os.getenv("QASA_LLM_MODEL", "qwen2.5-7b-instruct-local"),
        )
        self.max_retries = 1
        self.base_delay = 0.0
        if OpenAI is None:
            raise RuntimeError("openai package is required for vLLM mode")
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        self._embedding_fallback = _LocalLLM()

    def generate_content(self, model: str, contents: str, config: dict | None = None):
        config = config or {}
        schema = config.get("response_schema")
        messages = [{"role": "user", "content": contents}]
        if schema is not None:
            messages.insert(0, {
                "role": "system",
                "content": "Return only valid JSON matching the requested schema. Do not include markdown or explanations.",
            })
        request_kwargs = {}
        if os.getenv("QASA_DISABLE_THINKING", "").strip().lower() in {"1", "true", "yes"}:
            request_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }
        response = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=float(config.get("temperature", 0.0) or 0.0),
                    max_tokens=int(config.get("max_output_tokens", 512)),
                    **request_kwargs,
                )
                break
            except Exception as exc:
                message = str(exc).lower()
                retryable = any(
                    marker in message
                    for marker in ("429", "rate limit", "too many requests", "503", "unavailable", "timeout")
                )
                if not retryable or attempt + 1 >= self.max_retries:
                    raise
                time.sleep(self.base_delay * (2**attempt))
        if response is None:
            raise RuntimeError("OpenAI-compatible generation returned no response")
        text = response.choices[0].message.content or ""
        parsed = None
        if schema is not None:
            blob = _extract_json_blob(text)
            try:
                if hasattr(schema, "model_validate_json"):
                    parsed = schema.model_validate_json(blob)
                else:
                    parsed = schema.parse_raw(blob)
            except Exception:
                try:
                    data = json.loads(blob)
                    fields = getattr(schema, "model_fields", {})
                    if isinstance(data, dict) and "triples" in fields:
                        for triple in data.get("triples", []):
                            if isinstance(triple, dict) and "relation" not in triple and "predicate" in triple:
                                triple["relation"] = triple["predicate"]
                    if isinstance(data, dict):
                        parsed = schema.model_validate(data)
                    if isinstance(data, list) and "entities" in fields:
                        parsed = schema.model_validate({"entities": data})
                    elif isinstance(data, list) and "triples" in fields:
                        parsed = schema.model_validate({"triples": data})
                except Exception:
                    parsed = None
        return _LocalResponse(
            text=text,
            parsed=parsed,
            usage_metadata=_UsageMetadata(
                prompt_token_count=getattr(response.usage, "prompt_tokens", 0) or 0,
                candidates_token_count=getattr(response.usage, "completion_tokens", 0) or 0,
                total_token_count=getattr(response.usage, "total_tokens", 0) or 0,
            ),
        )

    def embed_content(self, model: str, contents: str | list[str]):
        return self._embedding_fallback.embed_content(model=model, contents=contents)


class _VLLMClient:
    def __init__(self) -> None:
        self.models = _VLLMModels()


class _QwenAPIModels(_VLLMModels):
    def __init__(self) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is required for the Qwen API backend")
        self.base_url = (
            os.getenv("QASA_QWEN_BASE_URL")
            or os.getenv("DASHSCOPE_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.api_key = os.getenv("QASA_QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Set QASA_QWEN_API_KEY (or DASHSCOPE_API_KEY) before using QASA_BACKEND=qwen_api"
            )
        self.model = os.getenv("QASA_QWEN_MODEL", "qwen-plus")
        self.max_retries = int(os.getenv("QASA_QWEN_MAX_RETRIES", "5"))
        self.base_delay = float(os.getenv("QASA_QWEN_RETRY_BASE_DELAY", "2"))
        timeout = float(os.getenv("QASA_QWEN_TIMEOUT", "120"))
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=timeout)
        self._embedding_fallback = _LocalLLM()


class _QwenAPIClient:
    def __init__(self) -> None:
        self.models = _QwenAPIModels()


class _GeminiModels:
    def __init__(self) -> None:
        if genai is None:
            raise RuntimeError("google-genai is required for the Gemini backend")
        api_key = (
            os.getenv("QASA_GEMINI_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "Set QASA_GEMINI_API_KEY (or GEMINI_API_KEY) before using QASA_BACKEND=gemini"
            )
        self.model = os.getenv("QASA_GEMINI_MODEL", "gemini-2.5-flash")
        self.max_retries = int(os.getenv("QASA_GEMINI_MAX_RETRIES", "5"))
        self.base_delay = float(os.getenv("QASA_GEMINI_RETRY_BASE_DELAY", "2"))
        self._client = genai.Client(api_key=api_key)

        self._embedding_fallback = _LocalLLM()

    def generate_content(self, model: str, contents: str, config: dict | None = None):
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config or {},
                )
            except Exception as exc:
                last_error = exc
                message = str(exc).lower()
                retryable = any(
                    marker in message
                    for marker in ("429", "resource exhausted", "too many requests", "503", "unavailable")
                )
                if not retryable or attempt + 1 >= self.max_retries:
                    raise
                time.sleep(self.base_delay * (2**attempt))
        raise RuntimeError("Gemini generation failed") from last_error

    def embed_content(self, model: str, contents: str | list[str]):
        return self._embedding_fallback.embed_content(model=model, contents=contents)


class _GeminiClient:
    def __init__(self) -> None:
        self.models = _GeminiModels()


def selected_backend() -> str:
    backend = os.getenv("QASA_BACKEND", "").strip().lower()
    aliases = {
        "google": "gemini",
        "google-genai": "gemini",
        "openai": "vllm",
        "qwen": "qwen_api",
        "qwen-api": "qwen_api",
        "dashscope": "qwen_api",
        "huggingface": "local",
    }
    if backend:
        return aliases.get(backend, backend)
    if _truthy(os.getenv("QASA_USE_VLLM")):
        return "vllm"
    if _truthy(os.getenv("QASA_USE_LOCAL_MODELS")) or os.getenv("QASA_LLM_MODEL"):
        return "local"
    return "gemini"


def configured_generation_model(default: str = "gemini-2.5-flash") -> str:
    backend = selected_backend()
    if backend == "gemini":
        return os.getenv("QASA_GEMINI_MODEL", default)
    if backend == "qwen_api":
        return os.getenv("QASA_QWEN_MODEL", "qwen-plus")
    if backend == "vllm":
        return os.getenv("QASA_VLLM_MODEL", os.getenv("QASA_LLM_MODEL", default))
    return os.getenv("QASA_LLM_MODEL", default)


def _should_use_local_models() -> bool:
    return selected_backend() == "local"


def _should_use_vllm() -> bool:
    return selected_backend() == "vllm"


_LOCAL_CLIENT: _LocalClient | None = None


def create_genai_client():
    global _LOCAL_CLIENT
    if selected_backend() == "gemini":
        return _GeminiClient()
    if selected_backend() == "qwen_api":
        return _QwenAPIClient()
    if _should_use_vllm():
        return _VLLMClient()
    if _should_use_local_models():
        if _LOCAL_CLIENT is None:
            _LOCAL_CLIENT = _LocalClient()
        return _LOCAL_CLIENT

    raise ValueError(
        f"Unsupported QASA_BACKEND={selected_backend()!r}; "
        "use gemini, qwen_api, vllm, or local"
    )
