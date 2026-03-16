"""Tests for the discovery layer."""

import os
from pathlib import Path

from dejavu.discovery import discover_files, discover_repos


class TestDiscoverRepos:
    def test_finds_git_repo(self, tmp_path):
        repo = tmp_path / "my-project"
        repo.mkdir()
        (repo / ".git").mkdir()
        (repo / "main.py").touch()

        repos = discover_repos([str(tmp_path)])
        assert repo in repos

    def test_finds_package_json_repo(self, tmp_path):
        repo = tmp_path / "js-app"
        repo.mkdir()
        (repo / "package.json").write_text("{}")

        repos = discover_repos([str(tmp_path)])
        assert repo in repos

    def test_skips_nonexistent_root(self, tmp_path):
        repos = discover_repos([str(tmp_path / "nonexistent")])
        assert repos == []

    def test_respects_max_depth(self, tmp_path):
        # Create deeply nested repo
        deep = tmp_path
        for i in range(10):
            deep = deep / f"level{i}"
            deep.mkdir()
        (deep / ".git").mkdir()

        repos = discover_repos([str(tmp_path)], max_depth=3)
        assert deep not in repos


class TestDiscoverFiles:
    def test_finds_python_files(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "utils.py").write_text("def util(): pass")

        files = discover_files(tmp_path)
        names = {f.name for f in files}
        assert "main.py" in names
        assert "utils.py" in names

    def test_skips_node_modules(self, tmp_path):
        nm = tmp_path / "node_modules" / "lodash"
        nm.mkdir(parents=True)
        (nm / "index.js").write_text("module.exports = {}")
        (tmp_path / "app.js").write_text("const x = 1;")

        files = discover_files(tmp_path)
        names = {f.name for f in files}
        assert "index.js" not in names
        assert "app.js" in names

    def test_skips_binary_extensions(self, tmp_path):
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "app.py").write_text("x = 1")

        files = discover_files(tmp_path)
        names = {f.name for f in files}
        assert "image.png" not in names
        assert "app.py" in names

    def test_skips_large_files(self, tmp_path):
        big = tmp_path / "huge.py"
        big.write_text("x" * (600 * 1024))  # 600KB

        files = discover_files(tmp_path, max_file_size=500 * 1024)
        assert big not in files

    def test_respects_gitignore(self, tmp_path):
        (tmp_path / ".gitignore").write_text("build/\n*.log\n")
        build = tmp_path / "build"
        build.mkdir()
        (build / "output.js").write_text("compiled")
        (tmp_path / "debug.log").write_text("log data")
        (tmp_path / "app.py").write_text("x = 1")

        files = discover_files(tmp_path)
        names = {f.name for f in files}
        assert "output.js" not in names
        assert "debug.log" not in names
        assert "app.py" in names

    def test_skips_hidden_files(self, tmp_path):
        (tmp_path / ".secret").write_text("hidden")
        (tmp_path / "visible.py").write_text("x = 1")

        files = discover_files(tmp_path)
        names = {f.name for f in files}
        assert ".secret" not in names
        assert "visible.py" in names
