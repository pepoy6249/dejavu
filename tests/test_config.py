"""Tests for the config layer."""

from pathlib import Path

from dejavu.config import DejavuConfig


class TestDejavuConfig:
    def test_defaults(self):
        config = DejavuConfig()
        assert config.embedding_model == "nomic-embed-code"
        assert config.ollama_base_url == "http://localhost:11434"
        assert config.default_limit == 10
        assert len(config.root_paths) > 0

    def test_save_and_load(self, tmp_path):
        config = DejavuConfig()
        config.root_paths = ["~/mycode"]
        config.embedding_model = "custom-model"
        config_path = tmp_path / "config.toml"
        config.save(config_path)

        loaded = DejavuConfig.load(config_path)
        assert loaded.root_paths == ["~/mycode"]
        assert loaded.embedding_model == "custom-model"

    def test_load_nonexistent_returns_defaults(self, tmp_path):
        config = DejavuConfig.load(tmp_path / "nope.toml")
        assert config.embedding_model == "nomic-embed-code"

    def test_env_override_db(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DEJAVU_DB", str(tmp_path / "custom.db"))
        config = DejavuConfig.load(tmp_path / "nope.toml")
        assert config.db_path == tmp_path / "custom.db"

    def test_env_override_ollama(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://remote:11434")
        config = DejavuConfig.load(tmp_path / "nope.toml")
        assert config.ollama_base_url == "http://remote:11434"
