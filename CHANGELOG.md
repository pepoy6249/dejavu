# Changelog

## 0.1.0 - Initial release

- Semantic code search via natural language descriptions
- Tree-sitter AST parsing for 20+ languages
- Local embeddings via Ollama (nomic-embed-code)
- SQLite storage with sqlite-vec KNN search (numpy fallback)
- CLI with search, index, status, config, and init commands
- MCP server for Claude integration
- Temporal hints ("last summer", "2024") and language filters
- Keyword boosting for improved relevance
- Incremental indexing (only re-processes modified files)
- .gitignore-aware file discovery
