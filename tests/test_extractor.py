"""Tests for the code extraction layer."""

from pathlib import Path

import pytest

try:
    from dejavu.extractor import CodeChunk, extract_chunks, is_indexable
except ImportError:
    pytest.skip("tree-sitter-languages not available", allow_module_level=True)


class TestIsIndexable:
    def test_python_file(self):
        assert is_indexable(Path("foo.py")) is True

    def test_javascript_file(self):
        assert is_indexable(Path("app.js")) is True

    def test_typescript_file(self):
        assert is_indexable(Path("app.ts")) is True

    def test_rust_file(self):
        assert is_indexable(Path("main.rs")) is True

    def test_random_binary(self):
        assert is_indexable(Path("image.png")) is False

    def test_dockerfile(self):
        assert is_indexable(Path("Dockerfile")) is True

    def test_makefile(self):
        assert is_indexable(Path("Makefile")) is True

    def test_sql_file(self):
        assert is_indexable(Path("schema.sql")) is True

    def test_css_file(self):
        assert is_indexable(Path("styles.css")) is True


class TestExtractChunksPython:
    def test_extracts_function(self, tmp_path):
        code = tmp_path / "example.py"
        code.write_text(
            'def hello_world(name: str) -> str:\n'
            '    """Greet someone."""\n'
            '    greeting = f"Hello, {name}!"\n'
            '    return greeting\n'
        )
        chunks = extract_chunks(code)
        assert len(chunks) >= 1
        # Should contain hello_world either as AST-extracted function or window
        assert any("hello_world" in c.source for c in chunks)

    def test_extracts_class(self, tmp_path):
        code = tmp_path / "models.py"
        code.write_text(
            'class UserManager:\n'
            '    """Manages users."""\n'
            '\n'
            '    def create_user(self, name: str):\n'
            '        """Create a new user."""\n'
            '        return {"name": name}\n'
            '\n'
            '    def delete_user(self, user_id: int):\n'
            '        """Delete a user by ID."""\n'
            '        pass\n'
        )
        chunks = extract_chunks(code)
        assert len(chunks) >= 1
        # Source should contain the class regardless of extraction method
        assert any("UserManager" in c.source for c in chunks)

    def test_empty_file_returns_nothing(self, tmp_path):
        code = tmp_path / "empty.py"
        code.write_text("")
        chunks = extract_chunks(code)
        assert chunks == []

    def test_whitespace_only_returns_nothing(self, tmp_path):
        code = tmp_path / "blank.py"
        code.write_text("   \n\n  \n")
        chunks = extract_chunks(code)
        assert chunks == []


class TestExtractChunksJavaScript:
    def test_extracts_function_declaration(self, tmp_path):
        code = tmp_path / "app.js"
        code.write_text(
            'function calculateTotal(items) {\n'
            '    let total = 0;\n'
            '    for (const item of items) {\n'
            '        total += item.price;\n'
            '    }\n'
            '    return total;\n'
            '}\n'
        )
        chunks = extract_chunks(code)
        assert len(chunks) >= 1
        assert any("calculateTotal" in c.source for c in chunks)

    def test_extracts_arrow_function(self, tmp_path):
        code = tmp_path / "utils.js"
        code.write_text(
            'const formatDate = (date) => {\n'
            '    const year = date.getFullYear();\n'
            '    const month = date.getMonth() + 1;\n'
            '    const day = date.getDate();\n'
            '    return `${year}-${month}-${day}`;\n'
            '};\n'
        )
        chunks = extract_chunks(code)
        assert len(chunks) >= 1
        assert any("formatDate" in c.source for c in chunks)


class TestSlidingWindowFallback:
    def test_sql_uses_sliding_window(self, tmp_path):
        code = tmp_path / "schema.sql"
        lines = [f"-- line {i}" for i in range(10)]
        lines.extend([
            "CREATE TABLE users (",
            "    id INTEGER PRIMARY KEY,",
            "    name TEXT NOT NULL",
            ");",
        ])
        code.write_text("\n".join(lines))
        chunks = extract_chunks(code)
        assert len(chunks) >= 1
        assert chunks[0].chunk_type == "window"
        assert chunks[0].language == "sql"

    def test_large_file_creates_multiple_windows(self, tmp_path):
        code = tmp_path / "big.sql"
        lines = [f"-- SQL comment line {i}" for i in range(200)]
        code.write_text("\n".join(lines))
        chunks = extract_chunks(code)
        assert len(chunks) > 1
        for c in chunks:
            assert c.chunk_type == "window"

    def test_nonexistent_file_returns_empty(self):
        chunks = extract_chunks(Path("/nonexistent/file.py"))
        assert chunks == []


class TestChunkMetadata:
    def test_chunks_have_line_numbers(self, tmp_path):
        code = tmp_path / "lines.py"
        code.write_text(
            '# comment\n'
            'def foo():\n'
            '    x = 1\n'
            '    y = 2\n'
            '    return x + y\n'
        )
        chunks = extract_chunks(code)
        assert len(chunks) >= 1
        for c in chunks:
            assert c.start_line >= 1
            assert c.end_line >= c.start_line

    def test_chunks_have_language(self, tmp_path):
        code = tmp_path / "test.py"
        code.write_text(
            'def bar():\n'
            '    return 42\n'
            '    # done\n'
        )
        chunks = extract_chunks(code)
        for c in chunks:
            assert c.language == "python"
