
"""
Repository ingestion orchestration service.

This module coordinates the repository scanning and parsing
process. It discovers source files, resolves the appropriate
language parser, extracts code definitions and aggregates
repository-level ingestion statistics.

The service is independent of individual parsing
implementations and follows the Parser Registry abstraction.
"""

from pathlib import Path

from src.ingestion.repository_reader import scan_repository
from src.parsers.parser_registry import ParserRegistry


class IngestionService:
    """
    Orchestrates the ingestion of a source code repository.

    Responsibilities:
        1. Scan the repository for supported files.
        2. Resolve the appropriate parser for each file.
        3. Extract normalized code definitions.
        4. Aggregate ingestion statistics.
        5. Capture file-level parsing failures.

    The service delegates file discovery and language-specific
    parsing to their respective components.
    """

    def __init__(
        self,
        parser_registry: ParserRegistry | None = None,
    ):
        """
        Initialize the ingestion service.

        Args:
            parser_registry:
                Optional parser registry dependency.
                If not supplied, a default registry is created.

        Dependency injection allows us to replace the registry
        during testing without modifying this service.
        """
        self.parser_registry = parser_registry or ParserRegistry()

    def ingest_repository(self, repository_path: str) -> dict:
        """
        Scan and parse an entire source code repository.

        Args:
            repository_path:
                Absolute or relative path to the repository.

        Returns:
            A dictionary containing:
                - Repository metadata.
                - File processing statistics.
                - Extracted code units.
                - File-level parsing errors.

        Raises:
            ValueError:
                If the repository path is invalid.
        """

        # Step 1: Resolve the repository path.
        repository_root = (
            Path(repository_path)
            .expanduser()
            .resolve()
        )

        # Step 2: Discover eligible files.
        files = scan_repository(str(repository_root))

        # Step 3: Initialize ingestion results.
        code_units = []

        parsed_files = 0
        unsupported_files = 0
        failed_files = 0

        errors = []

        # Step 4: Process each discovered file.
        for file in files:

            file_path = file["file_path"]

            # Resolve the language-specific parser.
            parser = self.parser_registry.get_parser(file_path)

            # No parser available for this file extension.
            if parser is None:
                unsupported_files += 1
                continue

            try:
                # Step 5: Parse the source code.
                definitions = parser.parse(
                    source_code=file["content"],
                    file_path=file_path,
                )

                # Step 6: Aggregate extracted definitions.
                code_units.extend(definitions)

                parsed_files += 1

            except Exception as exc:
                # Capture individual file failures without
                # terminating the entire repository ingestion.
                failed_files += 1

                errors.append({
                    "file_path": file_path,
                    "error": str(exc),
                })

        # Step 7: Return consolidated ingestion results.
        return {
            "repository": repository_root.name,
            "repository_path": str(repository_root),
            "total_files": len(files),
            "parsed_files": parsed_files,
            "unsupported_files": unsupported_files,
            "failed_files": failed_files,
            "total_code_units": len(code_units),
            "code_units": code_units,
            "errors": errors,
        }
