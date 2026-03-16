"""Extraction layer — tree-sitter AST parsing to extract meaningful code chunks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tree_sitter_languages import get_language, get_parser

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class CodeChunk:
    """A single extracted unit of code (function, class, method, etc.)."""
    chunk_type: str          # 'function', 'class', 'method', 'component', 'window'
    name: Optional[str]      # nullable for window chunks
    signature: Optional[str]
    docstring: Optional[str]
    source: str
    language: str
    start_line: int          # 1-indexed
    end_line: int            # 1-indexed


# ---------------------------------------------------------------------------
# Language config
# ---------------------------------------------------------------------------

# Maps file extension to (tree-sitter language name, node types to extract)
LANGUAGE_MAP: dict[str, tuple[str, list[str]]] = {
    ".py": ("python", [
        "function_definition",
        "class_definition",
    ]),
    ".js": ("javascript", [
        "function_declaration",
        "arrow_function",
        "class_declaration",
        "export_statement",
    ]),
    ".jsx": ("javascript", [
        "function_declaration",
        "arrow_function",
        "class_declaration",
        "export_statement",
    ]),
    ".ts": ("typescript", [
        "function_declaration",
        "arrow_function",
        "class_declaration",
        "export_statement",
    ]),
    ".tsx": ("tsx", [
        "function_declaration",
        "arrow_function",
        "class_declaration",
        "export_statement",
    ]),
    ".rs": ("rust", [
        "function_item",
        "impl_item",
        "struct_item",
        "enum_item",
    ]),
    ".go": ("go", [
        "function_declaration",
        "method_declaration",
        "type_declaration",
    ]),
    ".rb": ("ruby", [
        "method",
        "class",
        "module",
    ]),
    ".java": ("java", [
        "method_declaration",
        "class_declaration",
    ]),
    ".kt": ("kotlin", [
        "function_declaration",
        "class_declaration",
    ]),
    ".c": ("c", [
        "function_definition",
        "struct_specifier",
    ]),
    ".cpp": ("cpp", [
        "function_definition",
        "class_specifier",
        "struct_specifier",
    ]),
    ".h": ("c", [
        "function_definition",
        "struct_specifier",
    ]),
    ".hpp": ("cpp", [
        "function_definition",
        "class_specifier",
        "struct_specifier",
    ]),
    ".php": ("php", [
        "function_definition",
        "method_declaration",
        "class_declaration",
    ]),
    ".sh": ("bash", [
        "function_definition",
    ]),
    ".bash": ("bash", [
        "function_definition",
    ]),
    ".swift": ("swift", [
        "function_declaration",
        "class_declaration",
    ]),
}

# Extensions we recognize as code but don't have tree-sitter grammars for
FALLBACK_CODE_EXTENSIONS = {
    ".toml", ".yaml", ".yml", ".json", ".xml", ".ini", ".cfg",
    ".sql", ".graphql", ".gql", ".proto", ".tf", ".hcl",
    ".lua", ".vim", ".el", ".clj", ".cljs", ".erl", ".ex", ".exs",
    ".r", ".R", ".jl", ".scala", ".zig", ".nim", ".v",
    ".css", ".scss", ".less", ".html", ".svelte", ".vue",
    ".md", ".rst", ".txt",
    ".dockerfile", ".Dockerfile",
}

# Language name derived from extension for display
EXTENSION_TO_LANG: dict[str, str] = {
    ext: info[0] for ext, info in LANGUAGE_MAP.items()
}
# Add fallbacks
EXTENSION_TO_LANG.update({
    ".toml": "toml", ".yaml": "yaml", ".yml": "yaml",
    ".json": "json", ".xml": "xml", ".sql": "sql",
    ".graphql": "graphql", ".proto": "protobuf",
    ".css": "css", ".scss": "scss", ".html": "html",
    ".svelte": "svelte", ".vue": "vue",
    ".md": "markdown", ".rst": "rst",
    ".lua": "lua", ".r": "r", ".R": "r",
    ".jl": "julia", ".scala": "scala", ".zig": "zig",
    ".ex": "elixir", ".exs": "elixir",
})


# ---------------------------------------------------------------------------
# Name extraction helpers
# ---------------------------------------------------------------------------

def _get_node_name(node, language: str) -> Optional[str]:
    """Try to extract a name from a tree-sitter node."""
    # Most languages use a 'name' field
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf-8", errors="replace")

    # For arrow functions, look for parent variable declaration
    if node.type == "arrow_function" and node.parent:
        p = node.parent
        # const Foo = () => ...
        if p.type == "variable_declarator":
            n = p.child_by_field_name("name")
            if n:
                return n.text.decode("utf-8", errors="replace")
        # export default or export const
        if p.type == "export_statement":
            for child in p.children:
                if child.type == "lexical_declaration":
                    for sub in child.children:
                        if sub.type == "variable_declarator":
                            n = sub.child_by_field_name("name")
                            if n:
                                return n.text.decode("utf-8", errors="replace")

    # For export_statement, dig into the declaration
    if node.type == "export_statement":
        for child in node.children:
            if hasattr(child, "child_by_field_name"):
                n = child.child_by_field_name("name")
                if n:
                    return n.text.decode("utf-8", errors="replace")

    return None


def _get_signature(node, source_lines: list[str]) -> Optional[str]:
    """Extract function signature (first line or up to opening brace/colon)."""
    start = node.start_point[0]
    end = min(start + 5, node.end_point[0], len(source_lines))
    for i in range(start, end):
        line = source_lines[i]
        if any(c in line for c in (":", "{", "=>")):
            return "\n".join(source_lines[start : i + 1]).strip()
    return source_lines[start].strip() if start < len(source_lines) else None


def _get_docstring(node, source_lines: list[str], language: str) -> Optional[str]:
    """Extract docstring/leading comment from a node."""
    # Python: first child that's a string expression
    if language == "python":
        body = node.child_by_field_name("body")
        if body and body.child_count > 0:
            first = body.children[0]
            if first.type == "expression_statement" and first.child_count > 0:
                s = first.children[0]
                if s.type == "string":
                    return s.text.decode("utf-8", errors="replace").strip("\"'")

    # JS/TS/Rust/Go/etc: look for comment immediately before the node
    start_line = node.start_point[0]
    if start_line > 0:
        prev_line = source_lines[start_line - 1].strip()
        if prev_line.startswith("//") or prev_line.startswith("/*") or prev_line.startswith("///") or prev_line.startswith("#"):
            # Gather contiguous comment block
            comments = []
            i = start_line - 1
            while i >= 0:
                l = source_lines[i].strip()
                if l.startswith("//") or l.startswith("///") or l.startswith("#") or l.startswith("*") or l.startswith("/*"):
                    comments.insert(0, l.lstrip("/#* "))
                    i -= 1
                else:
                    break
            if comments:
                return "\n".join(comments)

    return None


def _classify_chunk_type(node_type: str) -> str:
    """Map tree-sitter node type to our chunk_type enum."""
    if "class" in node_type or "struct" in node_type or "enum" in node_type:
        return "class"
    if "method" in node_type or "impl" in node_type:
        return "method"
    if "module" in node_type:
        return "class"
    return "function"


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_chunks(file_path: Path) -> list[CodeChunk]:
    """
    Extract code chunks from a file using tree-sitter AST parsing.
    Falls back to sliding window for unsupported languages.
    """
    ext = file_path.suffix.lower()

    # Special case: Dockerfile has no extension
    if file_path.name.lower() in ("dockerfile", "makefile", "cmakelists.txt"):
        ext = f".{file_path.name.lower()}"

    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError) as e:
        log.debug(f"Cannot read {file_path}: {e}")
        return []

    if not source.strip():
        return []

    source_lines = source.split("\n")

    # Try tree-sitter first
    if ext in LANGUAGE_MAP:
        lang_name, node_types = LANGUAGE_MAP[ext]
        try:
            return _extract_with_treesitter(
                file_path, source, source_lines, lang_name, node_types
            )
        except Exception as e:
            log.debug(f"tree-sitter failed for {file_path}: {e}")
            # Fall through to sliding window

    # Sliding window fallback
    language = EXTENSION_TO_LANG.get(ext, ext.lstrip(".") or "unknown")
    if ext in FALLBACK_CODE_EXTENSIONS or ext in LANGUAGE_MAP:
        return _extract_sliding_window(file_path, source, source_lines, language)

    return []


def _extract_with_treesitter(
    file_path: Path,
    source: str,
    source_lines: list[str],
    lang_name: str,
    node_types: list[str],
) -> list[CodeChunk]:
    """Extract chunks using tree-sitter AST parsing."""
    language = get_language(lang_name)
    parser = get_parser(lang_name)

    tree = parser.parse(source.encode("utf-8"))
    chunks: list[CodeChunk] = []

    # Node types that are class-like containers — we extract the class AND its methods
    _CLASS_LIKE = {"class_definition", "class_declaration", "class_specifier",
                   "impl_item", "module"}
    # Node types that are method/function-like inside classes
    _METHOD_LIKE = {"function_definition", "function_declaration", "method_declaration",
                    "method", "arrow_function", "function_item"}

    def walk(node, inside_class: bool = False):
        if node.type in node_types:
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            node_source = "\n".join(source_lines[start_line : end_line + 1])

            # Skip tiny fragments (< 3 lines)
            if end_line - start_line < 2:
                return

            name = _get_node_name(node, lang_name)
            signature = _get_signature(node, source_lines)
            docstring = _get_docstring(node, source_lines, lang_name)
            chunk_type = _classify_chunk_type(node.type)

            # For export statements, only include if they contain a real declaration
            if node.type == "export_statement" and not name:
                # Still walk children
                for child in node.children:
                    walk(child)
                return

            chunks.append(CodeChunk(
                chunk_type=chunk_type,
                name=name,
                signature=signature,
                docstring=docstring,
                source=node_source,
                language=lang_name,
                start_line=start_line + 1,  # 1-indexed
                end_line=end_line + 1,
            ))

            # For class-like nodes, ALSO extract individual methods inside
            # so each method gets its own embedding (better search granularity)
            if node.type in _CLASS_LIKE:
                for child in node.children:
                    walk(child, inside_class=True)
            return

        # When inside a class body, extract methods that aren't in node_types
        if inside_class and node.type in _METHOD_LIKE:
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            if end_line - start_line >= 2:
                node_source = "\n".join(source_lines[start_line : end_line + 1])
                name = _get_node_name(node, lang_name)
                signature = _get_signature(node, source_lines)
                docstring = _get_docstring(node, source_lines, lang_name)
                chunks.append(CodeChunk(
                    chunk_type="method",
                    name=name,
                    signature=signature,
                    docstring=docstring,
                    source=node_source,
                    language=lang_name,
                    start_line=start_line + 1,
                    end_line=end_line + 1,
                ))
            return

        for child in node.children:
            walk(child, inside_class)

    walk(tree.root_node)

    # If tree-sitter found nothing meaningful, fall back to sliding window
    if not chunks:
        return _extract_sliding_window(file_path, source, source_lines, lang_name)

    return chunks


def _extract_sliding_window(
    file_path: Path,
    source: str,
    source_lines: list[str],
    language: str,
) -> list[CodeChunk]:
    """Sliding window fallback for files tree-sitter can't parse well."""
    chunks = []
    total = len(source_lines)

    if total <= 60:
        # Entire file as one chunk
        chunks.append(CodeChunk(
            chunk_type="window",
            name=file_path.name,
            signature=None,
            docstring=None,
            source=source,
            language=language,
            start_line=1,
            end_line=total,
        ))
    else:
        window = 50
        overlap = 10
        min_window = 20  # Don't create windows smaller than this
        i = 0
        window_num = 0
        while i < total:
            remaining = total - i
            # If the remaining lines would produce a tiny last window,
            # extend the current window to include everything
            if remaining <= window + min_window:
                end = total
            else:
                end = i + window
            chunk_source = "\n".join(source_lines[i:end])
            window_num += 1
            chunks.append(CodeChunk(
                chunk_type="window",
                name=f"{file_path.name}:window_{window_num}",
                signature=None,
                docstring=None,
                source=chunk_source,
                language=language,
                start_line=i + 1,
                end_line=end,
            ))
            if end >= total:
                break
            i += window - overlap

    return chunks


def is_indexable(file_path: Path) -> bool:
    """Check if a file should be indexed based on extension."""
    ext = file_path.suffix.lower()
    if ext in LANGUAGE_MAP or ext in FALLBACK_CODE_EXTENSIONS:
        return True
    if file_path.name.lower() in ("dockerfile", "makefile", "cmakelists.txt"):
        return True
    return False
