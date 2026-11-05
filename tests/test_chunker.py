
"""
Unit tests for the code-aware chunking module.

Validates symbol preservation, oversized symbol splitting,
line overlap, metadata preservation and deterministic IDs.
"""

from src.indexing.chunker import CodeChunker


def create_code_unit(content, start_line=1):
    """Create a representative normalized code unit."""

    return {
        "symbol_name": "calculate_total",
        "symbol_type": "function",
        "parent_symbol": "OrderService",
        "file_path": "services/order.py",
        "start_line": start_line,
        "end_line": start_line + len(content.splitlines()) - 1,
        "content": content,
        "language": "python",
    }


def test_small_symbol_remains_single_chunk():
    """Small functions should not be unnecessarily split."""

    unit = create_code_unit(
        "def hello():\n    return 'world'\n"
    )

    chunker = CodeChunker(max_chars=100)

    chunks = chunker.chunk_code_units([unit])

    assert len(chunks) == 1
    assert chunks[0]["content"] == unit["content"]
    assert chunks[0]["chunk_index"] == 0


def test_large_symbol_is_split():
    """Large functions should produce multiple chunks."""

    content = "".join(
        f"    variable_{i} = {i}\n"
        for i in range(30)
    )

    unit = create_code_unit(content)

    chunker = CodeChunker(
        max_chars=100,
        overlap_lines=2,
    )

    chunks = chunker.chunk_code_units([unit])

    assert len(chunks) > 1


def test_chunk_metadata_is_preserved():
    """Chunks must retain their original source metadata."""

    unit = create_code_unit(
        "def hello():\n    return 'world'\n"
    )

    chunker = CodeChunker()

    chunks = chunker.chunk_code_units([unit])

    chunk = chunks[0]

    assert chunk["symbol_name"] == "calculate_total"
    assert chunk["file_path"] == "services/order.py"
    assert chunk["language"] == "python"
    assert chunk["parent_symbol"] == "OrderService"


def test_chunk_ids_are_deterministic():
    """Identical source content should produce identical IDs."""

    unit = create_code_unit(
        "def hello():\n    return 'world'\n"
    )

    chunker = CodeChunker()

    first = chunker.chunk_code_units([unit])
    second = chunker.chunk_code_units([unit])

    assert first[0]["chunk_id"] == second[0]["chunk_id"]


def test_empty_code_units():
    """Empty input should return an empty chunk collection."""

    chunker = CodeChunker()

    assert chunker.chunk_code_units([]) == []


def test_empty_content_is_ignored():
    """Empty code definitions should not generate chunks."""

    unit = create_code_unit("")

    chunker = CodeChunker()

    assert chunker.chunk_code_units([unit]) == []


def test_invalid_configuration():
    """Invalid chunking parameters should be rejected."""

    try:
        CodeChunker(max_chars=0)
        assert False, "Expected ValueError"
    except ValueError:
        pass
