
"""
Common parser contract for RepoAnalyser AI.

All language-specific parsers must follow this interface so that
the ingestion and indexing pipelines can process source code
without depending on individual parser implementations.
"""

from abc import ABC, abstractmethod


class BaseParser(ABC):
    """Abstract base class for language-specific parsers."""

    @abstractmethod
    def parse(self, source_code: str, file_path: str) -> list[dict]:
        """
        Parse source code and return normalized code definitions.

        Args:
            source_code: Complete source file contents.
            file_path: Repository-relative source file path.

        Returns:
            A list of normalized code-unit dictionaries.
        """
        raise NotImplementedError
