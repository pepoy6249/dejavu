"""Config management — TOML-based configuration for Déjà Vu."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_DIR = Path.home() / ".dejavu"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.toml"
DEFAULT_DB_PATH = DEFAULT_CONFIG_DIR / "index.db"


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class DejavuConfig:
    """Runtime configuration for Déjà Vu."""

    # Paths to scan for code repos
    root_paths: list[str] = field(default_factory=lambda: [
        "~/code",
        "~/projects",
        "~/dev",
        "~/src",
        "~/repos",
        "~/work",
    ])

    # Path aliases for search filters
    path_aliases: dict[str, str] = field(default_factory=dict)

    # Database
    db_path: Path = DEFAULT_DB_PATH

    # Embedding
    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-code"
    embedding_fallback_model: str = "nomic-embed-text"
    embedding_batch_size: int = 32
    ollama_base_url: str = "http://localhost:11434"

    # Indexing
    max_file_size_kb: int = 500
    skip_dirs: list[str] = field(default_factory=lambda: [
        "node_modules", "venv", ".venv", "__pycache__", "dist", "build", "target", ".cache"
    ])
    skip_extensions: list[str] = field(default_factory=lambda: [
        ".min.js", ".map", ".lock", ".sum"
    ])

    # Search
    default_limit: int = 10
    keyword_boost: float = 0.15

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "DejavuConfig":
        """Load config from TOML file, falling back to defaults."""
        path = config_path or DEFAULT_CONFIG_PATH
        config = cls()

        if path.exists():
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib  # type: ignore
                except ImportError:
                    return config

            try:
                with open(path, "rb") as f:
                    data = tomllib.load(f)

                # Paths section
                paths = data.get("paths", {})
                if "roots" in paths:
                    config.root_paths = paths["roots"]
                if "aliases" in paths:
                    config.path_aliases = paths["aliases"]

                # Index section
                index = data.get("index", {})
                if "db_path" in index:
                    config.db_path = Path(index["db_path"]).expanduser()
                if "max_file_size_kb" in index:
                    config.max_file_size_kb = index["max_file_size_kb"]

                # Embedding section
                emb = data.get("embedding", {})
                if "provider" in emb:
                    config.embedding_provider = emb["provider"]
                if "model" in emb:
                    config.embedding_model = emb["model"]
                if "fallback_model" in emb:
                    config.embedding_fallback_model = emb["fallback_model"]
                if "batch_size" in emb:
                    config.embedding_batch_size = emb["batch_size"]

                ollama = emb.get("ollama", {})
                if "base_url" in ollama:
                    config.ollama_base_url = ollama["base_url"]

                # Search section
                search = data.get("search", {})
                if "default_limit" in search:
                    config.default_limit = search["default_limit"]
                if "keyword_boost" in search:
                    config.keyword_boost = search["keyword_boost"]

            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Error parsing {path}: {e}, using defaults")

        # Environment variable overrides
        if env_db := os.environ.get("DEJAVU_DB"):
            config.db_path = Path(env_db)
        if env_ollama := os.environ.get("OLLAMA_HOST"):
            config.ollama_base_url = env_ollama

        return config

    @staticmethod
    def _toml_list(items: list[str]) -> str:
        """Format a list of strings as a valid TOML array."""
        escaped = ['"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"' for s in items]
        return "[" + ", ".join(escaped) + "]"

    def save(self, config_path: Optional[Path] = None) -> None:
        """Save current config to TOML file."""
        path = config_path or DEFAULT_CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "[paths]",
            f"roots = {self._toml_list(self.root_paths)}",
            "",
        ]

        if self.path_aliases:
            lines.append("[paths.aliases]")
            for k, v in self.path_aliases.items():
                lines.append(f'{k} = "{v}"')
            lines.append("")

        lines.extend([
            "[index]",
            f'db_path = "{self.db_path}"',
            f"max_file_size_kb = {self.max_file_size_kb}",
            "",
            "[embedding]",
            f'provider = "{self.embedding_provider}"',
            f'model = "{self.embedding_model}"',
            f'fallback_model = "{self.embedding_fallback_model}"',
            f"batch_size = {self.embedding_batch_size}",
            "",
            "[embedding.ollama]",
            f'base_url = "{self.ollama_base_url}"',
            "",
            "[search]",
            f"default_limit = {self.default_limit}",
            f"keyword_boost = {self.keyword_boost}",
        ])

        path.write_text("\n".join(lines) + "\n")
