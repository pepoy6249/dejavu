# Dejavu

**Find code you forgot, by describing what it did.**

Dejavu is a semantic code search tool that lets you find code across your projects using natural language descriptions. Instead of remembering filenames, function names, or exact keywords, just describe what the code did:

```
dejavu "that drag and drop kanban board"
dejavu "CSV parser that grouped by date" --lang python
dejavu "animated sidebar component" --when "last summer"
```

## How it works

1. **Index** your code directories -- Dejavu uses tree-sitter AST parsing to extract functions, classes, and methods from 20+ languages
2. **Embed** each code chunk using local vector embeddings via [Ollama](https://ollama.com) (no data leaves your machine)
3. **Search** with natural language -- your query is embedded and matched against your code using vector similarity

Everything runs locally. Your code never leaves your machine.

## Install

```bash
pip install dejavu-code
```

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally

Pull the embedding model:

```bash
ollama pull nomic-embed-code
```

### Optional: faster vector search

For large codebases, install the sqlite-vec extension for hardware-accelerated KNN search:

```bash
pip install "dejavu-code[vec]"
```

Without it, Dejavu falls back to numpy-based cosine similarity (works fine for most codebases).

## Quick start

```bash
# 1. Initialize config
dejavu init

# 2. Edit ~/.dejavu/config.toml to set your code directories
#    (defaults: ~/code, ~/projects, ~/dev, ~/src, ~/repos, ~/work)

# 3. Index your code
dejavu index

# 4. Search!
dejavu "that function that parsed CSV files and grouped them by date"
```

### Index a specific directory

```bash
dejavu index ~/projects/my-app
```

### Filter by language or time

```bash
dejavu "auth middleware" --lang python
dejavu "React component with tabs" --when "last summer"
dejavu "deployment script" --path work
```

### Check index status

```bash
dejavu status
```

## Claude Code integration (MCP server)

Dejavu includes an [MCP](https://modelcontextprotocol.io) server that gives Claude direct access to your code search index. This is the primary way to use Dejavu -- Claude can find code you've written before without you needing to remember where it lives.

### Setup with Claude Code

Run this from your terminal:

```bash
claude mcp add dejavu -- dejavu-mcp
```

That's it. Claude Code will now have access to the `dejavu_search`, `dejavu_reindex`, `dejavu_status`, and `dejavu_forget` tools.

### Setup with Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "dejavu": {
      "command": "dejavu-mcp"
    }
  }
}
```

### What Claude can do with Dejavu

Once connected, you can ask Claude things like:

- "Search my code for that CSV parser I wrote last year"
- "Find the React component that had the animated sidebar"
- "Look for any auth middleware I wrote in Python"
- "Reindex my projects directory"

### Available MCP tools

| Tool | Description |
|------|-------------|
| `dejavu_search` | Search indexed code by natural language description. Supports language filters, temporal hints, and path filters. |
| `dejavu_reindex` | Index or re-index code directories. Incremental -- only processes modified files. |
| `dejavu_status` | Show index statistics: repo count, chunk count, languages, and configured paths. |
| `dejavu_forget` | Remove a repository/directory from the index. Source files are never modified. |

## Configuration

Config lives at `~/.dejavu/config.toml`. Created by `dejavu init`.

```toml
[paths]
roots = ["~/code", "~/projects"]

[index]
db_path = "~/.dejavu/index.db"
max_file_size_kb = 500

[embedding]
provider = "ollama"
model = "nomic-embed-code"
fallback_model = "nomic-embed-text"
batch_size = 32

[embedding.ollama]
base_url = "http://localhost:11434"

[search]
default_limit = 10
keyword_boost = 0.15
```

### Environment variable overrides

| Variable | Description |
|----------|-------------|
| `DEJAVU_DB` | Override database path |
| `OLLAMA_HOST` | Override Ollama URL |

## Supported languages

Tree-sitter AST parsing (extracts functions, classes, methods):

Python, JavaScript, TypeScript, TSX, Rust, Go, Ruby, Java, Kotlin, C, C++, PHP, Bash, Swift

Sliding-window fallback (indexes file contents in chunks):

SQL, HTML, CSS, SCSS, Svelte, Vue, TOML, YAML, JSON, Protobuf, Lua, Julia, Scala, Zig, Elixir, and more.

## Architecture

```
dejavu/
  cli.py         # Click CLI (search, index, status, config, init)
  server.py      # MCP server for Claude integration
  config.py      # TOML config loader
  db.py          # SQLite + sqlite-vec / numpy fallback
  embedder.py    # Ollama embedding client
  indexer.py     # Discovery -> extraction -> embedding pipeline
  extractor.py   # Tree-sitter AST parsing + sliding window fallback
  discovery.py   # Repo and file discovery (.gitignore aware)
  search.py      # Vector search + temporal/language hints + keyword boost
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[MIT](LICENSE)
