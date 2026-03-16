"""Tests for the database layer."""

import tempfile
from pathlib import Path

import pytest

from dejavu.db import DejavuDB, _serialize_f32, _deserialize_f32, EMBEDDING_DIM


@pytest.fixture
def db(tmp_path):
    """Create a temporary database for testing."""
    db = DejavuDB(tmp_path / "test.db")
    db.init_schema()
    yield db
    db.close()


class TestSerialization:
    def test_roundtrip(self):
        vec = [0.1, 0.2, 0.3, -0.5, 1.0]
        blob = _serialize_f32(vec)
        result = _deserialize_f32(blob)
        assert len(result) == len(vec)
        for a, b in zip(vec, result):
            assert abs(a - b) < 1e-6

    def test_empty_vector(self):
        vec = []
        blob = _serialize_f32(vec)
        assert _deserialize_f32(blob) == []


class TestRepos:
    def test_upsert_and_get(self, db):
        repo_id = db.upsert_repo("/home/user/project")
        assert repo_id > 0
        repo = db.get_repo("/home/user/project")
        assert repo is not None
        assert repo["path"] == "/home/user/project"

    def test_upsert_idempotent(self, db):
        id1 = db.upsert_repo("/home/user/project")
        id2 = db.upsert_repo("/home/user/project")
        assert id1 == id2

    def test_list_repos(self, db):
        db.upsert_repo("/a")
        db.upsert_repo("/b")
        repos = db.list_repos()
        assert len(repos) == 2
        paths = {r["path"] for r in repos}
        assert paths == {"/a", "/b"}

    def test_get_nonexistent(self, db):
        assert db.get_repo("/no/such/path") is None


class TestChunks:
    def test_insert_and_clear(self, db):
        repo_id = db.upsert_repo("/repo")
        chunk_id = db.insert_chunk(
            repo_id=repo_id,
            file_path="/repo/main.py",
            chunk_type="function",
            name="hello",
            signature="def hello():",
            docstring="Say hello.",
            source="def hello():\n    print('hi')\n    return True",
            language="python",
            start_line=1,
            end_line=3,
            file_mtime=1000.0,
        )
        assert chunk_id > 0

        cleared = db.clear_file_chunks("/repo/main.py")
        assert chunk_id in cleared

    def test_update_repo_counts(self, db):
        repo_id = db.upsert_repo("/repo")
        db.insert_chunk(
            repo_id=repo_id,
            file_path="/repo/a.py",
            chunk_type="function",
            name="f1",
            signature=None,
            docstring=None,
            source="def f1():\n    pass\n    pass",
            language="python",
            start_line=1,
            end_line=3,
            file_mtime=1000.0,
        )
        db.insert_chunk(
            repo_id=repo_id,
            file_path="/repo/b.py",
            chunk_type="function",
            name="f2",
            signature=None,
            docstring=None,
            source="def f2():\n    pass\n    pass",
            language="python",
            start_line=1,
            end_line=3,
            file_mtime=1000.0,
        )
        db.update_repo_counts(repo_id)
        repo = db.get_repo("/repo")
        assert repo["file_count"] == 2
        assert repo["chunk_count"] == 2


class TestStats:
    def test_empty_stats(self, db):
        stats = db.stats()
        assert stats["repos"] == 0
        assert stats["chunks"] == 0
        assert stats["embeddings"] == 0
        assert stats["languages"] == {}

    def test_stats_with_data(self, db):
        repo_id = db.upsert_repo("/repo")
        db.insert_chunk(
            repo_id=repo_id,
            file_path="/repo/main.py",
            chunk_type="function",
            name="foo",
            signature=None,
            docstring=None,
            source="def foo():\n    bar()\n    return True",
            language="python",
            start_line=1,
            end_line=3,
            file_mtime=1000.0,
        )
        stats = db.stats()
        assert stats["repos"] == 1
        assert stats["chunks"] == 1
        assert stats["languages"]["python"] == 1


class TestEmbeddings:
    def test_insert_and_search_numpy(self, db):
        # Force numpy fallback
        db.use_vec = False

        repo_id = db.upsert_repo("/repo")
        chunk_id = db.insert_chunk(
            repo_id=repo_id,
            file_path="/repo/main.py",
            chunk_type="function",
            name="hello",
            signature="def hello():",
            docstring=None,
            source="def hello():\n    print('hi')\n    return True",
            language="python",
            start_line=1,
            end_line=3,
            file_mtime=1000.0,
        )

        # Insert a simple embedding
        embedding = [0.1] * EMBEDDING_DIM
        db.insert_embedding(chunk_id, embedding)

        # Search with a similar query
        results = db.vector_search(
            query_embedding=[0.1] * EMBEDDING_DIM,
            limit=5,
        )
        assert len(results) >= 1
        assert results[0]["name"] == "hello"

    def test_search_with_language_filter(self, db):
        db.use_vec = False

        repo_id = db.upsert_repo("/repo")
        py_id = db.insert_chunk(
            repo_id=repo_id,
            file_path="/repo/main.py",
            chunk_type="function",
            name="py_func",
            signature=None,
            docstring=None,
            source="def py_func():\n    pass\n    pass",
            language="python",
            start_line=1,
            end_line=3,
            file_mtime=1000.0,
        )
        js_id = db.insert_chunk(
            repo_id=repo_id,
            file_path="/repo/app.js",
            chunk_type="function",
            name="js_func",
            signature=None,
            docstring=None,
            source="function js_func() {\n    return 1;\n    return 2;\n}",
            language="javascript",
            start_line=1,
            end_line=4,
            file_mtime=1000.0,
        )

        embedding = [0.1] * EMBEDDING_DIM
        db.insert_embedding(py_id, embedding)
        db.insert_embedding(js_id, embedding)

        results = db.vector_search(
            query_embedding=embedding,
            limit=5,
            language="python",
        )
        assert all(r["language"] == "python" for r in results)
