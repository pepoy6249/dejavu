"""Discovery layer — find code repositories and indexable files on disk."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pathspec

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Directories that signal "this is a code project root"
# Exact-match project root markers
PROJECT_MARKERS = {
    ".git",
    "package.json",
    "Cargo.toml",
    "pyproject.toml",
    "setup.py",
    "go.mod",
    "Makefile",
    "CMakeLists.txt",
    "pom.xml",
    "build.gradle",
    "Gemfile",
    "mix.exs",
    "dune-project",
    "stack.yaml",
}

# Extension-based project root markers (checked via suffix match)
PROJECT_MARKER_EXTENSIONS = {".sln"}

# Directories to always skip (never descend into)
SKIP_DIRS = {
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    ".git",
    "dist",
    "build",
    "target",
    ".cache",
    ".next",
    ".nuxt",
    ".output",
    ".turbo",
    "vendor",
    "deps",
    "_build",
    ".eggs",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "coverage",
    ".nyc_output",
    ".parcel-cache",
    ".webpack",
    "bower_components",
}

# File extensions to always skip
SKIP_EXTENSIONS = {
    ".min.js", ".min.css", ".map",
    ".lock", ".sum",
    ".wasm", ".pyc", ".pyo",
    ".so", ".dylib", ".dll", ".o", ".a",
    ".exe", ".bin",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".db", ".sqlite", ".sqlite3",
}

MAX_FILE_SIZE = 500 * 1024  # 500KB


# ---------------------------------------------------------------------------
# Gitignore parsing
# ---------------------------------------------------------------------------

def _load_gitignore(repo_path: Path) -> Optional[pathspec.PathSpec]:
    """Load .gitignore patterns from a repository root."""
    gitignore = repo_path / ".gitignore"
    if gitignore.exists():
        try:
            text = gitignore.read_text(encoding="utf-8", errors="replace")
            return pathspec.PathSpec.from_lines("gitignore", text.splitlines())
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Repo discovery
# ---------------------------------------------------------------------------

def discover_repos(root_paths: list[str], max_depth: int = 6) -> list[Path]:
    """
    Walk root paths to find code project directories.

    Returns a list of directory paths that contain project markers
    (e.g., .git, package.json, Cargo.toml).
    """
    repos: list[Path] = []
    seen: set[str] = set()

    for root in root_paths:
        root_path = Path(root).expanduser().resolve()
        if not root_path.is_dir():
            log.debug(f"Skipping non-directory root: {root}")
            continue

        _walk_for_repos(root_path, repos, seen, depth=0, max_depth=max_depth)

    return sorted(repos)


def _walk_for_repos(
    directory: Path,
    repos: list[Path],
    seen: set[str],
    depth: int,
    max_depth: int,
) -> None:
    """Recursively walk to find project roots."""
    if depth > max_depth:
        return

    dir_str = str(directory)
    if dir_str in seen:
        return
    seen.add(dir_str)

    # Check if current directory has any project markers
    try:
        entries = set(os.listdir(directory))
    except (PermissionError, OSError):
        return

    is_project = any(marker in entries for marker in PROJECT_MARKERS)
    if not is_project and PROJECT_MARKER_EXTENSIONS:
        is_project = any(
            e.endswith(ext) for e in entries for ext in PROJECT_MARKER_EXTENSIONS
        )

    if is_project:
        repos.append(directory)
        return  # Don't recurse deeper into project directories

    # Recurse into subdirectories
    _skip_dirs_lower = {s.lower() for s in SKIP_DIRS}
    for entry in entries:
        if entry.startswith(".") and entry != ".git":
            continue
        if entry.lower() in _skip_dirs_lower:
            continue
        if entry.endswith(".egg-info"):
            continue

        child = directory / entry
        if child.is_dir() and not child.is_symlink():
            _walk_for_repos(child, repos, seen, depth + 1, max_depth)


# ---------------------------------------------------------------------------
# File discovery within a repo
# ---------------------------------------------------------------------------

def discover_files(
    repo_path: Path,
    skip_extensions: Optional[set[str]] = None,
    max_file_size: int = MAX_FILE_SIZE,
) -> list[Path]:
    """
    Find all indexable files within a repo/directory.

    Respects .gitignore, skips binary/large files, and avoids SKIP_DIRS.
    """
    skip_ext = skip_extensions or SKIP_EXTENSIONS
    gitignore = _load_gitignore(repo_path)
    files: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(repo_path):
        current = Path(dirpath)

        # Prune skip directories in-place
        _skip_lower = {s.lower() for s in SKIP_DIRS}
        dirnames[:] = [
            d for d in dirnames
            if d.lower() not in _skip_lower
            and not d.startswith(".")
            and not d.endswith(".egg-info")
        ]

        for filename in filenames:
            file_path = current / filename
            rel_path = file_path.relative_to(repo_path)

            # Skip hidden files
            if filename.startswith("."):
                continue

            # Skip by extension
            ext = file_path.suffix.lower()
            if ext in skip_ext:
                continue

            # Skip large files
            try:
                if file_path.stat().st_size > max_file_size:
                    continue
            except OSError:
                continue

            # Respect .gitignore
            if gitignore and gitignore.match_file(str(rel_path)):
                continue

            files.append(file_path)

    return sorted(files)
