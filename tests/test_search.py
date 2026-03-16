"""Tests for the search engine layer."""

from dejavu.search import (
    compute_keyword_boost,
    parse_language_hint,
    parse_temporal_hint,
    strip_language_hint,
    strip_temporal_hint,
)


class TestParseTemporalHint:
    def test_explicit_year(self):
        result = parse_temporal_hint("something from 2024")
        assert result is not None
        start, end = result
        assert start < end

    def test_last_week(self):
        result = parse_temporal_hint("last week")
        assert result is not None
        start, end = result
        assert end > start

    def test_last_month(self):
        result = parse_temporal_hint("last month")
        assert result is not None

    def test_last_summer(self):
        result = parse_temporal_hint("last summer")
        assert result is not None

    def test_recently(self):
        result = parse_temporal_hint("recently")
        assert result is not None

    def test_no_temporal_hint(self):
        result = parse_temporal_hint("some random query about code")
        assert result is None


class TestStripTemporalHint:
    def test_strips_year(self):
        result = strip_temporal_hint("that parser from 2024")
        assert "2024" not in result
        assert "parser" in result

    def test_strips_last_week(self):
        result = strip_temporal_hint("auth middleware last week")
        assert "last week" not in result.lower()
        assert "auth" in result

    def test_preserves_non_temporal(self):
        text = "drag and drop kanban board"
        assert strip_temporal_hint(text) == text


class TestParseLanguageHint:
    def test_in_python(self):
        assert parse_language_hint("that CSV parser in python") == "python"

    def test_the_javascript_one(self):
        assert parse_language_hint("the javascript component") == "javascript"

    def test_written_in_rust(self):
        assert parse_language_hint("written in rust") == "rust"

    def test_using_typescript(self):
        assert parse_language_hint("using typescript") == "typescript"

    def test_no_language_hint(self):
        assert parse_language_hint("that function I wrote") is None

    def test_ambiguous_go_not_matched_without_context(self):
        # "go" is ambiguous -- should not match without "in go" context
        assert parse_language_hint("go ahead and search") is None

    def test_python_alias_py(self):
        assert parse_language_hint("my py script") == "python"


class TestStripLanguageHint:
    def test_strips_in_python(self):
        result = strip_language_hint("CSV parser in python")
        assert "python" not in result.lower()
        assert "CSV" in result

    def test_preserves_non_language(self):
        text = "drag and drop board"
        assert strip_language_hint(text) == text


class TestKeywordBoost:
    def test_full_match(self):
        chunk = {
            "name": "parse_csv",
            "signature": "def parse_csv(path: str)",
            "docstring": "Parse a CSV file and return rows.",
        }
        boost = compute_keyword_boost("parse csv file", chunk, boost_weight=0.15)
        assert boost > 0

    def test_no_match(self):
        chunk = {
            "name": "render_button",
            "signature": "function renderButton(props)",
            "docstring": "Render a button component.",
        }
        boost = compute_keyword_boost("database migration", chunk, boost_weight=0.15)
        assert boost == 0.0

    def test_partial_match(self):
        chunk = {
            "name": "parse_csv",
            "signature": "def parse_csv(path)",
            "docstring": None,
        }
        boost = compute_keyword_boost("parse json data", chunk, boost_weight=0.15)
        assert 0 < boost < 0.15

    def test_empty_query(self):
        chunk = {"name": "foo", "signature": None, "docstring": None}
        assert compute_keyword_boost("", chunk) == 0.0
