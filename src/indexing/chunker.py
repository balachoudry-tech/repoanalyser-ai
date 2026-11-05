
"""
Code-aware chunking module for RepoAnalyser AI.

Converts normalized code definitions into searchable chunks.

Small functions and classes are preserved as individual chunks.
Large definitions are split into smaller line-based chunks while
preserving source locations and symbol metadata.

The resulting chunks are intended for embedding generation
and vector database indexing.
"""

from hashlib import sha256


DEFAULT_MAX_CHARS = 4000
DEFAULT_OVERLAP_LINES = 5


class CodeChunker:
    """
    Splits extracted source code definitions into searchable chunks.

    The chunker uses parser-generated code units rather than
    arbitrary text boundaries.

    Attributes:
        max_chars: Maximum target character count per chunk.
        overlap_lines: Number of source lines repeated between
            consecutive chunks when splitting large symbols.
    """

    def __init__(
        self,
        max_chars: int = DEFAULT_MAX_CHARS,
        overlap_lines: int = DEFAULT_OVERLAP_LINES,
    ):
        """
        Initialize the code chunker.

        Args:
            max_chars: Maximum target characters per chunk.
            overlap_lines: Number of overlapping lines.

        Raises:
            ValueError: If chunk configuration is invalid.
        """
        if max_chars <= 0:
            raise ValueError("max_chars must be greater than zero")

        if overlap_lines < 0:
            raise ValueError("overlap_lines cannot be negative")

        self.max_chars = max_chars
        self.overlap_lines = overlap_lines

    def chunk_code_units(self, code_units: list[dict]) -> list[dict]:
        """
        Convert normalized code units into searchable chunks.

        Args:
            code_units: Code definitions extracted by language parsers.

        Returns:
            A list of code chunks with content and source metadata.
        """
        chunks = []

        for unit in code_units:

            content = unit.get("content", "")

            if not content or not content.strip():
                continue

            # Preserve complete symbols when they fit.
            if len(content) <= self.max_chars:
                chunks.append(
                    self._create_chunk(
                        unit=unit,
                        content=content,
                        start_line=unit["start_line"],
                        end_line=unit["end_line"],
                        chunk_index=0,
                    )
                )
                continue

            # Split oversized symbols while preserving source lines.
            chunks.extend(self._split_large_symbol(unit))

        return chunks

    def _split_large_symbol(self, unit: dict) -> list[dict]:
        """
        Split an oversized symbol into line-based chunks.

        Consecutive chunks share a configurable number of lines
        to preserve nearby source context.

        A single line exceeding max_chars is split by characters.
        Its source line number is retained.

        Args:
            unit: Normalized code definition.

        Returns:
            Smaller chunks containing the original source metadata.
        """
        lines = unit["content"].splitlines(keepends=True)

        chunks = []
        chunk_index = 0
        start_index = 0

        while start_index < len(lines):

            current_lines = []
            current_size = 0
            end_index = start_index

            # Build a chunk without exceeding the target size.
            while end_index < len(lines):

                line = lines[end_index]

                if current_lines and (
                    current_size + len(line) > self.max_chars
                ):
                    break

                if not current_lines and len(line) > self.max_chars:
                    # Handle an unusually long single source line.
                    current_lines.append(line[:self.max_chars])
                    current_size = self.max_chars
                    end_index += 1
                    break

                current_lines.append(line)
                current_size += len(line)
                end_index += 1

            chunk_content = "".join(current_lines)

            # Calculate original source line positions.
            chunk_start_line = unit["start_line"] + start_index
            chunk_end_line = (
                unit["start_line"] + end_index - 1
            )

            # A long individual line may have been sliced.
            if (
                end_index == start_index + 1
                and len(lines[start_index]) > self.max_chars
            ):
                chunk_end_line = chunk_start_line

            chunks.append(
                self._create_chunk(
                    unit=unit,
                    content=chunk_content,
                    start_line=chunk_start_line,
                    end_line=chunk_end_line,
                    chunk_index=chunk_index,
                )
            )

            chunk_index += 1

            if end_index >= len(lines):
                break

            # Move forward while retaining line overlap.
            next_start = max(
                start_index + 1,
                end_index - self.overlap_lines,
            )

            start_index = next_start

        return chunks

    def _create_chunk(
        self,
        unit: dict,
        content: str,
        start_line: int,
        end_line: int,
        chunk_index: int,
    ) -> dict:
        """
        Create a normalized searchable chunk.

        Metadata is copied from the original code unit so
        retrieval can identify the source file and symbol.

        A deterministic identifier is generated from the source
        location, chunk index and content.
        """

        chunk_identity = (
            f"{unit['file_path']}:"
            f"{unit['symbol_name']}:"
            f"{start_line}:"
            f"{end_line}:"
            f"{chunk_index}:"
            f"{content}"
        )

        chunk_id = sha256(
            chunk_identity.encode("utf-8")
        ).hexdigest()

        return {
            "chunk_id": chunk_id,
            "symbol_name": unit["symbol_name"],
            "symbol_type": unit["symbol_type"],
            "parent_symbol": unit.get("parent_symbol"),
            "file_path": unit["file_path"],
            "language": unit["language"],
            "start_line": start_line,
            "end_line": end_line,
            "chunk_index": chunk_index,
            "content": content,
        }
