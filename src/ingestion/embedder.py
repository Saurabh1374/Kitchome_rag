import os
import time
import json
import hashlib
import logging
import importlib
import urllib.request
import urllib.error
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from config import config as app_config, EmbedderConfig

logger = logging.getLogger(__name__)


@dataclass
class BatchEmbeddingResult:
    """
    Structured result returned by embed_batch, supporting both direct iteration
    over vectors (for backwards compatibility) and access to compute telemetry.
    """
    embeddings: List[List[float]]
    provider: str
    model: str
    dimension: int
    latency_ms: float = 0.0
    token_count: int = 0

    def __iter__(self):
        return iter(self.embeddings)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, index):
        return self.embeddings[index]


class BaseEmbedder(ABC):
    """
    Abstract contract for all embedding compute engines.
    Zero knowledge of database queues, SQLite locks, or cursor tracking.
    """
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generates a dense vector embedding for a single text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> BatchEmbeddingResult:
        """Generates dense vector embeddings for a batch of texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the dimensionality of the generated vectors."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns the identifier name of the provider."""
        pass

    def _l2_normalize(self, vector: np.ndarray) -> List[float]:
        """Utility method to perform L2 unit normalization."""
        norm = np.linalg.norm(vector)
        if norm > 1e-12:
            vector = vector / norm
        return vector.tolist()


class HashEmbedder(BaseEmbedder):
    """
    Zero-dependency deterministic MD5/n-gram hash projection engine.
    Used for fast offline testing, air-gapped CI, and zero-cost local execution.
    """
    def __init__(self, dimension: int = 384, normalize: bool = True):
        self._dimension = dimension
        self.normalize = normalize

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_name(self) -> str:
        return "hash"

    def embed_text(self, text: str) -> List[float]:
        text_clean = text.lower().strip()
        tokens = text_clean.split()
        vector = np.zeros(self._dimension, dtype=np.float32)

        if not tokens:
            vector[0] = 1.0
        else:
            for token in tokens:
                hash_val = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                idx = hash_val % self._dimension
                val = (hash_val % 100) / 100.0 - 0.5
                vector[idx] += val

                if len(token) >= 3:
                    for i in range(len(token) - 2):
                        sub = token[i:i + 3]
                        sub_hash = int(hashlib.md5(sub.encode("utf-8")).hexdigest(), 16)
                        sub_idx = sub_hash % self._dimension
                        vector[sub_idx] += 0.2

        if self.normalize:
            return self._l2_normalize(vector)
        return vector.tolist()

    def embed_batch(self, texts: List[str]) -> BatchEmbeddingResult:
        t0 = time.perf_counter()
        embeddings = [self.embed_text(t) for t in texts]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        total_tokens = sum(len(t.split()) for t in texts)
        return BatchEmbeddingResult(
            embeddings=embeddings,
            provider=self.provider_name,
            model="hash-projection",
            dimension=self.dimension,
            latency_ms=latency_ms,
            token_count=total_tokens
        )


class HuggingFaceEmbedder(BaseEmbedder):
    """
    First-class Hugging Face provider supporting both:
    1. Local inference via sentence-transformers (GPU / MPS / CPU)
    2. Serverless Inference API via HF_TOKEN (zero local PyTorch footprint)
    """
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        dimension: int = 384,
        device: str = "cpu",
        api_key: Optional[str] = None,
        is_api: bool = False,
        normalize: bool = True,
        timeout_seconds: float = 30.0
    ):
        self.model_name = model_name
        self._dimension = dimension
        self.device = device
        self.api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
        self.is_api = is_api
        self.normalize = normalize
        self.timeout_seconds = timeout_seconds
        self._local_model = None

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_name(self) -> str:
        return "huggingface_api" if self.is_api else "huggingface"

    def _get_local_model(self):
        if self._local_model is None:
            try:
                st_module = importlib.import_module("sentence_transformers")
                SentenceTransformer = getattr(st_module, "SentenceTransformer")
            except ImportError:
                raise ImportError(
                    "The 'sentence-transformers' package is required for local Hugging Face embeddings.\n"
                    "Install it using: pip install sentence-transformers"
                ) from None
            logger.info(f"Loading local Hugging Face model: {self.model_name} on device={self.device}")
            self._local_model = SentenceTransformer(self.model_name, device=self.device)
            if hasattr(self._local_model, "get_sentence_embedding_dimension"):
                detected_dim = self._local_model.get_sentence_embedding_dimension()
                if detected_dim:
                    self._dimension = int(detected_dim)
        return self._local_model

    def _embed_via_hf_api(self, texts: List[str]) -> List[List[float]]:
        """Invokes Hugging Face serverless Inference API over standard HTTPS."""
        if not self.api_key:
            raise ValueError(
                "A Hugging Face token is required for 'huggingface_api'. "
                "Set HF_TOKEN or HUGGINGFACE_API_KEY environment variable."
            )

        endpoint = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_name}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = json.dumps({"inputs": texts, "options": {"wait_for_model": True}}).encode("utf-8")

        req = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if isinstance(data, dict):
                    if "error" in data:
                        raise RuntimeError(f"Hugging Face API returned error: {data['error']}")
                    raise RuntimeError(f"Unexpected dictionary response from Hugging Face API: {data}")
                if texts and isinstance(data, list) and data and isinstance(data[0], (int, float)):
                    data = [data]
                if self.normalize:
                    data = [self._l2_normalize(np.array(v, dtype=np.float32)) for v in data]
                return data
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Hugging Face API error ({e.code}): {err_body}")

    def embed_text(self, text: str) -> List[float]:
        return self.embed_batch([text]).embeddings[0]

    def embed_batch(self, texts: List[str]) -> BatchEmbeddingResult:
        if not texts:
            return BatchEmbeddingResult(
                embeddings=[],
                provider=self.provider_name,
                model=self.model_name,
                dimension=self.dimension
            )

        t0 = time.perf_counter()
        if self.is_api:
            embeddings = self._embed_via_hf_api(texts)
        else:
            model = self._get_local_model()
            raw = model.encode(
                texts,
                batch_size=len(texts),
                device=self.device,
                normalize_embeddings=self.normalize,
                show_progress_bar=False
            )
            embeddings = [v.tolist() for v in raw]

        latency_ms = (time.perf_counter() - t0) * 1000.0
        total_tokens = sum(len(t.split()) for t in texts)

        return BatchEmbeddingResult(
            embeddings=embeddings,
            provider=self.provider_name,
            model=self.model_name,
            dimension=len(embeddings[0]) if embeddings else self.dimension,
            latency_ms=latency_ms,
            token_count=total_tokens
        )


class OllamaEmbedder(BaseEmbedder):
    """
    Zero-dependency local REST client for self-hosted Ollama server (e.g. nomic-embed-text).
    Uses standard library urllib.request with zero extra pip packages.
    """
    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        dimension: int = 768,
        normalize: bool = True,
        timeout_seconds: float = 30.0
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self._dimension = dimension
        self.normalize = normalize
        self.timeout_seconds = timeout_seconds

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_name(self) -> str:
        return "ollama"

    def embed_text(self, text: str) -> List[float]:
        endpoint = f"{self.base_url}/api/embeddings"
        payload = json.dumps({"model": self.model_name, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                vec = np.array(data.get("embedding", []), dtype=np.float32)
                if self.normalize:
                    return self._l2_normalize(vec)
                return vec.tolist()
        except Exception as e:
            raise RuntimeError(f"Ollama embedding request failed: {e}")

    def embed_batch(self, texts: List[str]) -> BatchEmbeddingResult:
        t0 = time.perf_counter()
        embeddings = [self.embed_text(t) for t in texts]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        total_tokens = sum(len(t.split()) for t in texts)
        actual_dim = len(embeddings[0]) if embeddings else self.dimension
        return BatchEmbeddingResult(
            embeddings=embeddings,
            provider=self.provider_name,
            model=self.model_name,
            dimension=actual_dim,
            latency_ms=latency_ms,
            token_count=total_tokens
        )


class OpenAIEmbedder(BaseEmbedder):
    """
    Cloud embeddings via OpenAI API (text-embedding-3-small, text-embedding-3-large).
    Supports custom dimension projection and uses standard library HTTP with retry.
    """
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        dimension: int = 1536,
        api_key: Optional[str] = None,
        normalize: bool = True,
        timeout_seconds: float = 30.0,
        max_retries: int = 3
    ):
        self.model_name = model_name
        self._dimension = dimension
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.normalize = normalize
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_name(self) -> str:
        return "openai"

    def embed_batch(self, texts: List[str]) -> BatchEmbeddingResult:
        if not self.api_key:
            raise ValueError("An OpenAI API key is required. Set OPENAI_API_KEY environment variable.")

        endpoint = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        body: Dict[str, Any] = {"model": self.model_name, "input": texts}
        if "text-embedding-3" in self.model_name:
            body["dimensions"] = self.dimension

        payload = json.dumps(body).encode("utf-8")
        last_err = None

        t0 = time.perf_counter()
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    data_items = sorted(res_json.get("data", []), key=lambda x: x.get("index", 0))
                    embeddings = [item["embedding"] for item in data_items]
                    if self.normalize:
                        embeddings = [self._l2_normalize(np.array(e, dtype=np.float32)) for e in embeddings]

                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    usage = res_json.get("usage") or {}
                    token_count = usage.get("total_tokens") or sum(len(t.split()) for t in texts)

                    return BatchEmbeddingResult(
                        embeddings=embeddings,
                        provider=self.provider_name,
                        model=self.model_name,
                        dimension=len(embeddings[0]) if embeddings else self.dimension,
                        latency_ms=latency_ms,
                        token_count=token_count
                    )
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="ignore")
                last_err = RuntimeError(f"OpenAI API error ({e.code}): {err_body}")
                if e.code in (429, 500, 503):
                    time.sleep(1.0 * (2 ** attempt))
                    continue
                raise last_err
            except Exception as e:
                last_err = e
                time.sleep(1.0 * (2 ** attempt))

        raise RuntimeError(f"OpenAI embedding failed after {self.max_retries} attempts: {last_err}")

    def embed_text(self, text: str) -> List[float]:
        return self.embed_batch([text]).embeddings[0]


class FastEmbedder(BaseEmbedder):
    """
    Ultra-fast ONNX CPU inference via fastembed.
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", dimension: int = 384, normalize: bool = True):
        self.model_name = model_name
        self._dimension = dimension
        self.normalize = normalize
        self._model = None

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_name(self) -> str:
        return "fastembed"

    def _get_model(self):
        if self._model is None:
            try:
                fe_module = importlib.import_module("fastembed")
                TextEmbedding = getattr(fe_module, "TextEmbedding")
            except ImportError:
                raise ImportError(
                    "The 'fastembed' package is required for FastEmbed ONNX embeddings.\n"
                    "Install it using: pip install fastembed"
                ) from None
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed_text(self, text: str) -> List[float]:
        return self.embed_batch([text]).embeddings[0]

    def embed_batch(self, texts: List[str]) -> BatchEmbeddingResult:
        if not texts:
            return BatchEmbeddingResult(
                embeddings=[],
                provider=self.provider_name,
                model=self.model_name,
                dimension=self.dimension
            )
        t0 = time.perf_counter()
        model = self._get_model()
        raw = list(model.embed(texts))
        embeddings = [v.tolist() for v in raw]
        if self.normalize:
            embeddings = [self._l2_normalize(np.array(e, dtype=np.float32)) for e in embeddings]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        total_tokens = sum(len(t.split()) for t in texts)
        return BatchEmbeddingResult(
            embeddings=embeddings,
            provider=self.provider_name,
            model=self.model_name,
            dimension=len(embeddings[0]) if embeddings else self.dimension,
            latency_ms=latency_ms,
            token_count=total_tokens
        )


def get_embedder(
    config: Optional[EmbedderConfig] = None,
    **overrides
) -> BaseEmbedder:
    """
    Factory function instantiating the configured embedding provider.
    Supports runtime overrides via kwargs.
    """
    cfg = config or getattr(app_config, "embedder", EmbedderConfig())
    raw_provider = overrides.get("provider") if overrides.get("provider") is not None else cfg.provider
    provider = str(raw_provider or "hash").lower().strip()
    model_name = overrides.get("model_name") if overrides.get("model_name") is not None else cfg.model_name
    dimension = overrides.get("dimension") if overrides.get("dimension") is not None else cfg.dimension
    normalize = overrides.get("normalize") if overrides.get("normalize") is not None else cfg.normalize
    device = overrides.get("device") if overrides.get("device") is not None else cfg.device
    api_key = overrides.get("api_key") if overrides.get("api_key") is not None else cfg.api_key
    base_url = overrides.get("base_url") if overrides.get("base_url") is not None else cfg.base_url
    timeout_seconds = overrides.get("timeout_seconds") if overrides.get("timeout_seconds") is not None else cfg.timeout_seconds

    if provider in ("hash", "mock"):
        return HashEmbedder(dimension=dimension, normalize=normalize)

    elif provider in ("huggingface", "sentence_transformers", "hf"):
        return HuggingFaceEmbedder(
            model_name=model_name,
            dimension=dimension,
            device=device,
            api_key=api_key,
            is_api=False,
            normalize=normalize,
            timeout_seconds=timeout_seconds
        )

    elif provider in ("huggingface_api", "hf_api"):
        return HuggingFaceEmbedder(
            model_name=model_name,
            dimension=dimension,
            device=device,
            api_key=api_key,
            is_api=True,
            normalize=normalize,
            timeout_seconds=timeout_seconds
        )

    elif provider == "ollama":
        return OllamaEmbedder(
            model_name=model_name,
            base_url=base_url,
            dimension=dimension,
            normalize=normalize,
            timeout_seconds=timeout_seconds
        )

    elif provider == "openai":
        return OpenAIEmbedder(
            model_name=model_name,
            dimension=dimension,
            api_key=api_key,
            normalize=normalize,
            timeout_seconds=timeout_seconds
        )

    elif provider == "fastembed":
        return FastEmbedder(
            model_name=model_name,
            dimension=dimension,
            normalize=normalize
        )

    else:
        logger.warning(f"Unknown embedding provider '{provider}'. Falling back to HashEmbedder.")
        return HashEmbedder(dimension=dimension, normalize=normalize)


class EmbeddingEngine(BaseEmbedder):
    """
    Drop-in backward-compatible wrapper around get_embedder().
    Preserves existing initialization signature: EmbeddingEngine(dimension=384).
    """
    def __init__(
        self,
        dimension: Optional[int] = None,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
    ):
        overrides: Dict[str, Any] = {}
        if dimension is not None:
            overrides["dimension"] = dimension
        if provider is not None:
            overrides["provider"] = provider
        if model_name is not None:
            overrides["model_name"] = model_name
        overrides.update(kwargs)

        self._embedder = get_embedder(**overrides)

    @property
    def dimension(self) -> int:
        return self._embedder.dimension

    @property
    def provider_name(self) -> str:
        return self._embedder.provider_name

    def embed_text(self, text: str) -> List[float]:
        return self._embedder.embed_text(text)

    def embed_batch(self, texts: List[str]) -> BatchEmbeddingResult:
        return self._embedder.embed_batch(texts)

