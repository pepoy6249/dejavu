"""Database layer — SQLite + sqlite-vec (or numpy fallback) for chunk storage and vector search."""

from __future__ import annotations

import logging
import sqlite3
import struct
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Try to load sqlite-vec; fall back to numpy KNN if unavailable
_HAS_SQLITE_VEC = False
try:
    import sqlite_vec
    _HAS_SQLITE_VEC = True
except Exception:
    pass

# numpy is only needed for the fallback search path
_np = None
def _get_numpy():
    """Lazy-load numpy only when needed (fallback search path)."""
    global _np
    if _np is None:
        try:
            import numpy as np
            _np = np
        except ImportError:
            raise ImportError(
                "numpy is required for vector search when sqlite-vec is unavailable. "
                "Install it with: pip install numpy"
            )
    return _np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 768  # nomic-embed-code output dimension

DEFAULT_DB_PATH = Path.home() / ".dejavu" / "index.db"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_f32(vec: list[float]) -> bytes:
    """Serialize a list of floats into a compact binary format for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


def _deserialize_f32(raw: bytes) -> list[float]:
    """Deserialize sqlite-vec binary back to float list."""
    n = len(raw) // 4
    return list(struct.unpack(f"{n}f", raw))


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

class DejavuDB:
    """Manages the Déjà Vu SQLite database with vector search.

    Uses sqlite-vec for KNN search when available. Falls back to a numpy-based
    brute-force cosine similarity search otherwise (still fast for <1M vectors).
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self.use_vec = _HAS_SQLITE_VEC

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row

            if self.use_vec:
                try:
                    self._conn.enable_load_extension(True)
                    sqlite_vec.load(self._conn)
                    self._conn.enable_load_extension(False)
                except Exception as e:
                    log.warning(f"sqlite-vec unavailable ({e}), using numpy fallback")
                    self.use_vec = False

            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def init_schema(self) -> None:
        """Create tables if they don't exist."""
        c = self.conn
        c.executescript("""
            CREATE TABLE IF NOT EXISTS repos (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                last_scan_at REAL NOT NULL,
                file_count INTEGER DEFAULT 0,
                chunk_count INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                repo_id INTEGER NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
                file_path TEXT NOT NULL,
                chunk_type TEXT NOT NULL,
                name TEXT,
                signature TEXT,
                docstring TEXT,
                source TEXT NOT NULL,
                language TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                file_mtime REAL NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
            CREATE INDEX IF NOT EXISTS idx_chunks_language ON chunks(language);
            CREATE INDEX IF NOT EXISTS idx_chunks_mtime ON chunks(file_mtime);
            CREATE INDEX IF NOT EXISTS idx_chunks_name ON chunks(name);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_unique
                ON chunks(file_path, chunk_type, COALESCE(name, ''), start_line);

            -- Fallback embeddings table (used when sqlite-vec is unavailable)
            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                vector BLOB NOT NULL
            );
        """)

        if self.use_vec:
            try:
                c.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                        chunk_id INTEGER PRIMARY KEY,
                        embedding float[{EMBEDDING_DIM}]
                    )
                """)
            except Exception as e:
                log.warning(f"Could not create vec_chunks table: {e}")
                self.use_vec = False

        c.commit()

    # ------------------------------------------------------------------
    # Repos
    # ------------------------------------------------------------------

    def upsert_repo(self, path: str) -> int:
        """Insert or update a repo, returning its id."""
        c = self.conn
        c.execute(
            """INSERT INTO repos (path, last_scan_at)
               VALUES (?, ?)
               ON CONFLICT(path) DO UPDATE SET last_scan_at=excluded.last_scan_at""",
            (path, time.time()),
        )
        c.commit()
        row = c.execute("SELECT id FROM repos WHERE path=?", (path,)).fetchone()
        return row["id"]

    def get_repo(self, path: str) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM repos WHERE path=?", (path,)).fetchone()
        return dict(row) if row else None

    def list_repos(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM repos ORDER BY path").fetchall()
        return [dict(r) for r in rows]

    def update_repo_counts(self, repo_id: int) -> None:
        c = self.conn
        file_count = c.execute(
            "SELECT COUNT(DISTINCT file_path) FROM chunks WHERE repo_id=?", (repo_id,)
        ).fetchone()[0]
        chunk_count = c.execute(
            "SELECT COUNT(*) FROM chunks WHERE repo_id=?", (repo_id,)
        ).fetchone()[0]
        c.execute(
            "UPDATE repos SET file_count=?, chunk_count=? WHERE id=?",
            (file_count, chunk_count, repo_id),
        )
        c.commit()

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------

    def clear_file_chunks(self, file_path: str) -> list[int]:
        """Delete all chunks for a file, returning their IDs (for vec cleanup)."""
        c = self.conn
        rows = c.execute(
            "SELECT id FROM chunks WHERE file_path=?", (file_path,)
        ).fetchall()
        ids = [r["id"] for r in rows]
        if ids:
            placeholders = ",".join("?" * len(ids))
            if self.use_vec:
                c.execute(f"DELETE FROM vec_chunks WHERE chunk_id IN ({placeholders})", ids)
            c.execute(f"DELETE FROM embeddings WHERE chunk_id IN ({placeholders})", ids)
            c.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", ids)
            c.commit()
        return ids

    def insert_chunk(
        self,
        repo_id: int,
        file_path: str,
        chunk_type: str,
        name: Optional[str],
        signature: Optional[str],
        docstring: Optional[str],
        source: str,
        language: str,
        start_line: int,
        end_line: int,
        file_mtime: float,
        _commit: bool = True,
    ) -> int:
        """Insert a chunk and return its ID.

        Set _commit=False when inserting many chunks in a batch,
        then call commit_batch() when done.
        """
        c = self.conn
        cur = c.execute(
            """INSERT INTO chunks
               (repo_id, file_path, chunk_type, name, signature, docstring,
                source, language, start_line, end_line, file_mtime, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                repo_id, file_path, chunk_type, name, signature, docstring,
                source, language, start_line, end_line, file_mtime, time.time(),
            ),
        )
        if _commit:
            c.commit()
        return cur.lastrowid

    def insert_embedding(self, chunk_id: int, embedding: list[float], _commit: bool = True) -> None:
        """Insert a vector embedding for a chunk.

        Set _commit=False when inserting many embeddings in a batch,
        then call commit_batch() when done.
        """
        c = self.conn
        blob = _serialize_f32(embedding)

        if self.use_vec:
            c.execute(
                "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, blob),
            )
        c.execute(
            "INSERT OR REPLACE INTO embeddings (chunk_id, vector) VALUES (?, ?)",
            (chunk_id, blob),
        )
        if _commit:
            c.commit()

    def commit_batch(self) -> None:
        """Commit a batch of inserts. Call after insert_chunk/_embedding with _commit=False."""
        self.conn.commit()

    def insert_embeddings_batch(self, pairs: list[tuple[int, list[float]]]) -> None:
        """Batch insert embeddings."""
        c = self.conn
        for chunk_id, embedding in pairs:
            blob = _serialize_f32(embedding)
            if self.use_vec:
                c.execute(
                    "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, blob),
                )
            c.execute(
                "INSERT OR REPLACE INTO embeddings (chunk_id, vector) VALUES (?, ?)",
                (chunk_id, blob),
            )
        c.commit()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def vector_search(
        self,
        query_embedding: list[float],
        limit: int = 50,
        language: Optional[str] = None,
        min_mtime: Optional[float] = None,
        max_mtime: Optional[float] = None,
        path_contains: Optional[str] = None,
    ) -> list[dict]:
        """
        KNN vector search, returning chunks sorted by similarity.

        Uses sqlite-vec when available, falls back to numpy brute-force.
        """
        if self.use_vec:
            return self._vec_search(
                query_embedding, limit, language, min_mtime, max_mtime, path_contains
            )
        return self._numpy_search(
            query_embedding, limit, language, min_mtime, max_mtime, path_contains
        )

    def _vec_search(
        self,
        query_embedding: list[float],
        limit: int,
        language: Optional[str],
        min_mtime: Optional[float],
        max_mtime: Optional[float],
        path_contains: Optional[str],
    ) -> list[dict]:
        """KNN search using sqlite-vec extension."""
        fetch_k = limit * 3 if (language or min_mtime or path_contains) else limit

        rows = self.conn.execute(
            """
            SELECT c.*, v.distance
            FROM vec_chunks v
            JOIN chunks c ON c.id = v.chunk_id
            WHERE v.embedding MATCH ?
                AND k = ?
            ORDER BY v.distance
            """,
            (_serialize_f32(query_embedding), fetch_k),
        ).fetchall()

        return self._apply_filters(rows, limit, language, min_mtime, max_mtime, path_contains)

    def _numpy_search(
        self,
        query_embedding: list[float],
        limit: int,
        language: Optional[str],
        min_mtime: Optional[float],
        max_mtime: Optional[float],
        path_contains: Optional[str],
    ) -> list[dict]:
        """Brute-force cosine similarity search using numpy.

        Processes embeddings in batches to limit peak memory usage.
        For very large indices (>100K chunks), sqlite-vec is strongly recommended.
        """
        np = _get_numpy()
        BATCH_SIZE = 10_000  # Process 10K embeddings at a time (~30MB per batch)
        fetch_k = limit * 3 if (language or min_mtime or path_contains) else limit
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = max(float(np.linalg.norm(query_vec)), 1e-10)
        query_vec_normed = query_vec / query_norm

        # Collect top-k candidates across all batches
        top_candidates: list[tuple[float, int]] = []  # (distance, chunk_id)

        cursor = self.conn.execute("SELECT chunk_id, vector FROM embeddings")
        while True:
            rows = cursor.fetchmany(BATCH_SIZE)
            if not rows:
                break

            chunk_ids = [r["chunk_id"] for r in rows]
            vectors = np.array(
                [_deserialize_f32(r["vector"]) for r in rows], dtype=np.float32
            )

            norms = np.linalg.norm(vectors, axis=1)
            norms = np.where(norms == 0, 1e-10, norms)
            similarities = (vectors @ query_vec_normed) / norms
            distances = 1.0 - similarities

            # Keep only the best fetch_k from this batch + previous candidates
            for i, cid in enumerate(chunk_ids):
                dist = float(distances[i])
                if len(top_candidates) < fetch_k:
                    top_candidates.append((dist, cid))
                elif dist < top_candidates[-1][0]:
                    top_candidates.append((dist, cid))

            # Trim to fetch_k after each batch
            top_candidates.sort(key=lambda x: x[0])
            top_candidates = top_candidates[:fetch_k]

        if not top_candidates:
            return []

        # Fetch chunk data for top results
        results = []
        for dist, cid in top_candidates:
            row = self.conn.execute("SELECT * FROM chunks WHERE id=?", (cid,)).fetchone()
            if row:
                d = dict(row)
                d["distance"] = dist
                results.append(d)

        return self._apply_filters(
            results, limit, language, min_mtime, max_mtime, path_contains,
            is_dict=True,
        )

    def _apply_filters(
        self,
        rows,
        limit: int,
        language: Optional[str],
        min_mtime: Optional[float],
        max_mtime: Optional[float],
        path_contains: Optional[str],
        is_dict: bool = False,
    ) -> list[dict]:
        """Apply post-search filters to results."""
        results = []
        for r in rows:
            d = r if is_dict else dict(r)
            if language and d["language"] != language:
                continue
            if min_mtime and d["file_mtime"] < min_mtime:
                continue
            if max_mtime and d["file_mtime"] > max_mtime:
                continue
            if path_contains and path_contains.lower() not in d["file_path"].lower():
                continue
            results.append(d)
            if len(results) >= limit:
                break
        return results

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        c = self.conn
        repos = c.execute("SELECT COUNT(*) FROM repos").fetchone()[0]
        chunks = c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        embeddings = c.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        languages = c.execute(
            "SELECT language, COUNT(*) as cnt FROM chunks GROUP BY language ORDER BY cnt DESC"
        ).fetchall()
        return {
            "repos": repos,
            "chunks": chunks,
            "embeddings": embeddings,
            "languages": {r["language"]: r["cnt"] for r in languages},
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
