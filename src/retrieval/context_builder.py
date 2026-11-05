
"""
Context construction for CodePilot AI.

Transforms retrieved source-code chunks into structured context
that can be supplied to a Large Language Model (LLM).

Responsibilities:
    - Remove duplicate retrieval results.
    - Preserve source-code metadata.
    - Generate source citation identifiers.
    - Enforce a maximum context character budget.
"""

import re
from typing import Any


class ContextBuilder:
    """
    Build structured LLM context from retrieved code chunks.

    The builder does not generate answers or perform retrieval.
    It prepares evidence returned by RetrievalService.
    """

    DEFAULT_MAX_CONTEXT_CHARS = 12000

    def __init__(
        self,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    ) -> None:
        """
        Initialize the context builder.

        Args:
            max_context_chars: Maximum number of characters allowed
                in the generated context.

        Raises:
            ValueError: If the configured limit is not positive.
        """
        if (
            not isinstance(max_context_chars, int)
            or isinstance(max_context_chars, bool)
            or max_context_chars <= 0
        ):
            raise ValueError("max_context_chars must be a positive integer.")

        self.max_context_chars = max_context_chars

    @staticmethod
    def _get_deduplication_key(result: dict[str, Any]) -> str:
        """
        Generate a stable key for identifying duplicate results.

        ChromaDB returns a unique vector ID for each stored chunk.
        If an ID is unavailable, use source metadata and content
        as a fallback.
        """
        vector_id = result.get("id")

        if vector_id:
            return str(vector_id)

        metadata = result.get("metadata") or {}

        return "|".join(
            [
                str(metadata.get("file_path", "")),
                str(metadata.get("start_line", "")),
                str(metadata.get("end_line", "")),
                str(result.get("content", "")),
            ]
        )

    @staticmethod
    def _format_source(
        source_id: str,
        result: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """
        Format one retrieved chunk and its citation metadata.

        Returns:
            A tuple containing the formatted context block and
            structured source metadata.
        """
        metadata = result.get("metadata") or {}

        file_path = str(metadata.get("file_path", "unknown"))
        symbol_name = str(metadata.get("symbol_name", "unknown"))
        language = str(metadata.get("language", "text"))

        start_line = metadata.get("start_line")
        end_line = metadata.get("end_line")

        content = result.get("content", "")

        # Restrict language labels to safe characters for Markdown fences.
        safe_language = (
            language if re.fullmatch(r"[a-zA-Z0-9_+-]{1,30}", language)
            else "text"
        )

        line_range = (
            f"{start_line}-{end_line}"
            if start_line is not None and end_line is not None
            else "unknown"
        )

        distance = result.get("distance")

        header = (
            f"[{source_id}]\n"
            f"File: {file_path}\n"
            f"Symbol: {symbol_name}\n"
            f"Lines: {line_range}\n"
        )

        if distance is not None:
            header += f"Cosine distance: {distance}\n"

        block = (
            f"{header}\n"
            f"```{safe_language}\n"
            f"{content}\n"
            f"```"
        )

        source = {
            "source_id": source_id,
            "vector_id": result.get("id"),
            "file_path": file_path,
            "symbol_name": symbol_name,
            "start_line": start_line,
            "end_line": end_line,
            "language": safe_language,
            "distance": distance,
        }

        return block, source

    def build_context(
        self,
        retrieval_result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Construct structured context from retrieval results.

        Results are processed in their retrieval order. This preserves
        the ranking produced by the vector search.

        Chunks that exceed the remaining character budget are skipped
        rather than truncated, preserving source-code integrity.

        Args:
            retrieval_result: Output dictionary returned by
                RetrievalService.retrieve().

        Returns:
            A dictionary containing:
                - context: Formatted context for the LLM.
                - sources: Metadata for generated citations.
                - source_count: Number of included sources.
                - skipped_count: Number of skipped chunks.
                - truncated: Whether any chunks were skipped.

        Raises:
            ValueError: If the retrieval result has an invalid structure.
        """
        if not isinstance(retrieval_result, dict):
            raise ValueError("Retrieval result must be a dictionary.")

        results = retrieval_result.get("results")

        if not isinstance(results, list):
            raise ValueError("Retrieval results must be a list.")

        if not results:
            return {
                "context": "No relevant source code was found.",
                "sources": [],
                "source_count": 0,
                "skipped_count": 0,
                "truncated": False,
            }

        context_blocks = []
        sources = []
        seen_ids = set()

        current_length = 0
        skipped_count = 0

        for result in results:
            if not isinstance(result, dict):
                raise ValueError("Every retrieval result must be a dictionary.")

            deduplication_key = self._get_deduplication_key(result)

            # Ignore chunks already processed.
            if deduplication_key in seen_ids:
                continue

            seen_ids.add(deduplication_key)

            content = result.get("content")

            if not isinstance(content, str) or not content.strip():
                skipped_count += 1
                continue

            source_id = f"S{len(sources) + 1}"

            block, source = self._format_source(
                source_id=source_id,
                result=result,
            )

            additional_length = len(block)

            if context_blocks:
                additional_length += 2

            # Never truncate a source-code chunk.
            if current_length + additional_length > self.max_context_chars:
                skipped_count += 1
                continue

            context_blocks.append(block)
            sources.append(source)
            current_length += additional_length

        context = "\n\n".join(context_blocks)

        if not context:
            context = "No relevant source code could fit within the context limit."

        return {
            "context": context,
            "sources": sources,
            "source_count": len(sources),
            "skipped_count": skipped_count,
            "truncated": skipped_count > 0,
        }
