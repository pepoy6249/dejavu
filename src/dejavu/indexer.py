"""Indexing pipeline — Orchestrates discovery, extraction, embedding, and storage."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from .config import DejavuConfig
from .db import DejavuDB
from .discovery import discover_files, discover_repos
from .embedder import OllamaEmbedder, prepare_document_text
from .extractor import CodeChunk, extract_chunks, is_indexable

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Progress callback type
# ---------------------------------------------------------------------------

class IndexProgress:
    """Tracks indexing progress."""
    def __init__(self):
        self.repos_found: int = 0
        self.repos_indexed: int = 0
        self.files_found: int = 0
        self.files_indexed: int = 0
        self.chunks_extracted: int = 0
        self.chunks_embedded: int = 0
        self.errors: list[str] = []
        self.current_repo: str = ""
        self.current_file: str = ""


# ---------------------------------------------------------------------------
# Main indexing function
# ---------------------------------------------------------------------------

async def index_path(
    path: str,
    config: DejavuConfig,
    db: DejavuDB,
    embedder: OllamaEmbedder,
    progress: Optional[IndexProgress] = None,
) -> IndexProgress:
    """
    Index a single directory (repo or arbitrary path).

    1. Discover indexable files
    2. Extract code chunks via tree-sitter
    3. Embed chunks in batches
    4. Store in database
    """
    if progress is None:
        progress = IndexProgress()

    repo_path = Path(path).expanduser().resolve()
    if not repo_path.is_dir():
        progress.errors.append(f"Not a directory: {path}")
        return progress

    progress.current_repo = str(repo_path)

    # Get existing repo info BEFORE upsert overwrites last_scan_at
    repo_info = db.get_repo(str(repo_path))
    last_scan = repo_info.get("last_scan_at", 0) if repo_info else 0

    # Now upsert (updates last_scan_at to current time)
    repo_id = db.upsert_repo(str(repo_path))

    # Discover files
    files = discover_files(
        repo_path,
        max_file_size=config.max_file_size_kb * 1024,
    )
    progress.files_found += len(files)

    # Filter to indexable files and check mtime for incremental
    indexable_files = []
    for f in files:
        if not is_indexable(f):
            continue
        try:
            mtime = f.stat().st_mtime
            # Only re-index files modified since last scan
            # (skip this check on first index when last_scan is 0)
            if last_scan > 0 and mtime < last_scan:
                continue
            indexable_files.append((f, mtime))
        except OSError:
            continue

    if not indexable_files:
        log.info(f"No new/modified files in {repo_path}")
        db.update_repo_counts(repo_id)
        progress.repos_indexed += 1
        return progress

    # Extract and embed
    pending_chunks: list[tuple[int, str]] = []  # (chunk_id, text_for_embedding)

    for file_path, mtime in indexable_files:
        progress.current_file = str(file_path)

        try:
            # Clear old chunks for this file
            db.clear_file_chunks(str(file_path))

            # Extract chunks
            chunks = extract_chunks(file_path)
            if not chunks:
                continue

            progress.files_indexed += 1

            for chunk in chunks:
                # Store chunk in DB (deferred commit for performance)
                chunk_id = db.insert_chunk(
                    repo_id=repo_id,
                    file_path=str(file_path),
                    chunk_type=chunk.chunk_type,
                    name=chunk.name,
                    signature=chunk.signature,
                    docstring=chunk.docstring,
                    source=chunk.source,
                    language=chunk.language,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    file_mtime=mtime,
                    _commit=False,
                )

                # Prepare text for embedding
                embed_text = prepare_document_text(
                    language=chunk.language,
                    chunk_type=chunk.chunk_type,
                    name=chunk.name,
                    signature=chunk.signature,
                    docstring=chunk.docstring,
                    source=chunk.source,
                )
                pending_chunks.append((chunk_id, embed_text))
                progress.chunks_extracted += 1

        except Exception as e:
            log.warning(f"Error processing {file_path}: {e}")
            progress.errors.append(f"{file_path}: {e}")

    # Commit all chunk inserts in one transaction
    db.commit_batch()

    # Batch embed all chunks
    if pending_chunks:
        chunk_ids = [c[0] for c in pending_chunks]
        texts = [c[1] for c in pending_chunks]

        try:
            embeddings = await embedder.embed_batch(
                texts, batch_size=config.embedding_batch_size
            )

            pairs = list(zip(chunk_ids, embeddings))
            db.insert_embeddings_batch(pairs)
            progress.chunks_embedded += len(pairs)

        except Exception as e:
            log.error(f"Embedding failed: {e}")
            progress.errors.append(f"Embedding error: {e}")

    db.update_repo_counts(repo_id)
    progress.repos_indexed += 1
    return progress


async def index_all(
    config: DejavuConfig,
    db: DejavuDB,
    embedder: OllamaEmbedder,
    progress: Optional[IndexProgress] = None,
) -> IndexProgress:
    """
    Full index: discover repos from configured root paths, then index each.
    """
    if progress is None:
        progress = IndexProgress()

    repos = discover_repos(config.root_paths)
    progress.repos_found = len(repos)

    for repo in repos:
        try:
            await index_path(
                str(repo), config, db, embedder, progress
            )
        except Exception as e:
            log.error(f"Error indexing {repo}: {e}")
            progress.errors.append(f"{repo}: {e}")

    return progress
