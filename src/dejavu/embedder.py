"""Embedding layer — Generate vector embeddings via Ollama or fallback providers."""

from __future__ import annotations

import logging
from typing import Optional

import httpx

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "nomic-embed-code"
FALLBACK_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768


# ---------------------------------------------------------------------------
# Ollama Embedder
# ---------------------------------------------------------------------------

class OllamaEmbedder:
    """
    Generate embeddings using Ollama's local API.

    Uses nomic-embed-code by default, which supports task-prefixed prompts:
    - 'search_document: ...' for indexing chunks
    - 'search_query: ...' for search queries
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_MODEL,
        fallback_model: str = FALLBACK_MODEL,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.fallback_model = fallback_model
        self._active_model: Optional[str] = None

    async def _check_model(self, client: httpx.AsyncClient, model: str) -> bool:
        """Check if a model is available in Ollama."""
        try:
            resp = await client.get(f"{self.base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
            models = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
            return model in models
        except Exception:
            return False

    async def _resolve_model(self, client: httpx.AsyncClient) -> str:
        """Find the best available model."""
        if self._active_model:
            return self._active_model

        if await self._check_model(client, self.model):
            self._active_model = self.model
        elif await self._check_model(client, self.fallback_model):
            log.warning(
                f"{self.model} not found, falling back to {self.fallback_model}"
            )
            self._active_model = self.fallback_model
        else:
            raise RuntimeError(
                f"No embedding model available in Ollama. "
                f"Please run: ollama pull {self.model}"
            )
        return self._active_model

    async def embed_one(self, text: str) -> list[float]:
        """Embed a single text string."""
        async with httpx.AsyncClient() as client:
            model = await self._resolve_model(client)
            resp = await client.post(
                f"{self.base_url}/api/embed",
                json={"model": model, "input": text},
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            # Ollama returns {"embeddings": [[...]]} for /api/embed
            embeddings = data.get("embeddings", [])
            if embeddings:
                return embeddings[0]
            raise RuntimeError(f"Empty embedding response: {data}")

    async def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        Embed multiple texts in batches.

        Ollama's /api/embed supports batch input natively.
        """
        all_embeddings: list[list[float]] = []

        async with httpx.AsyncClient() as client:
            model = await self._resolve_model(client)

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                resp = await client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": model, "input": batch},
                    timeout=120.0,
                )
                resp.raise_for_status()
                data = resp.json()
                embeddings = data.get("embeddings", [])
                if len(embeddings) != len(batch):
                    raise RuntimeError(
                        f"Expected {len(batch)} embeddings, got {len(embeddings)}"
                    )
                all_embeddings.extend(embeddings)

        return all_embeddings

    async def is_available(self) -> bool:
        """Check if Ollama is running and has an embedding model."""
        try:
            async with httpx.AsyncClient() as client:
                await self._resolve_model(client)
                return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------

def prepare_document_text(
    language: str,
    chunk_type: str,
    name: Optional[str],
    signature: Optional[str],
    docstring: Optional[str],
    source: str,
    max_source_lines: int = 200,
) -> str:
    """
    Prepare a code chunk for embedding as a document.

    Constructs a composite string that gives the embedding model
    rich context about what the chunk does:
      search_document: [language] [type] [name]
      [signature]
      [docstring]
      [source preview]
    """
    parts = [f"search_document: {language} {chunk_type}"]

    if name:
        parts[0] += f" {name}"

    if signature:
        parts.append(signature)

    if docstring:
        parts.append(docstring[:500])  # cap docstring length

    # Truncate source to max lines
    source_lines = source.split("\n")
    if len(source_lines) > max_source_lines:
        source = "\n".join(source_lines[:max_source_lines])

    parts.append(source)

    return "\n".join(parts)


def prepare_query_text(query: str) -> str:
    """
    Prepare a search query for embedding.

    Prefixes with 'search_query:' as expected by nomic-embed-code.
    """
    return f"search_query: {query}"
