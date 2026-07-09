"""Tests for rag/store.py's embedding function construction.

Previously _init() always built an OllamaEmbeddingFunction regardless of
model.provider — RAG/knowledge search silently returned nothing under
openai_compatible (LM Studio, ...), since it was hitting Ollama's /api/embed
shape against a server that doesn't speak it. Now that openai_compatible is
the only provider, _build_embedding_function always builds an
OpenAIEmbeddingFunction pointed at model.base_url."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.store import _build_embedding_function


def _cfg(base_url, model, api_key=""):
    return SimpleNamespace(model_base_url=base_url, model_api_key=api_key, embedding_model=model)


def test_builds_openai_embedding_function():
    cfg = _cfg("http://localhost:1234/v1", "text-embedding-nomic-embed-text-v1.5")
    fn = _build_embedding_function(cfg)
    assert type(fn).__name__ == "OpenAIEmbeddingFunction"


def test_blank_api_key_uses_placeholder():
    # Must not raise for a local server (LM Studio) with no auth configured.
    cfg = _cfg("http://localhost:1234/v1", "some-embed-model", api_key="")
    fn = _build_embedding_function(cfg)
    assert type(fn).__name__ == "OpenAIEmbeddingFunction"


def test_strips_trailing_slash_from_base_url():
    cfg = _cfg("http://localhost:1234/v1/", "some-embed-model")
    fn = _build_embedding_function(cfg)
    assert fn.api_base == "http://localhost:1234/v1"


def test_passes_through_model_name():
    cfg = _cfg("http://localhost:1234/v1", "text-embedding-nomic-embed-text-v1.5")
    fn = _build_embedding_function(cfg)
    assert fn.model_name == "text-embedding-nomic-embed-text-v1.5"
