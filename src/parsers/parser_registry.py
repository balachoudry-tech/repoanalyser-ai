
from pathlib import Path

from src.parsers.base import BaseParser
from src.parsers.python_parser import PythonParser
from src.parsers.tree_sitter_parser import TreeSitterParser


class ParserRegistry:
    """
    Registry responsible for resolving language-specific parsers.

    Maps file extensions to reusable parser instances.
    """

    def __init__(self):
        self._parsers: dict[str, BaseParser] = {
            ".py": PythonParser(),
            ".js": TreeSitterParser("javascript"),
            ".jsx": TreeSitterParser("javascript"),
            ".ts": TreeSitterParser("typescript"),
            ".tsx": TreeSitterParser("typescript"),
            ".go": TreeSitterParser("go"),
        }

    def get_parser(self, file_path: str) -> BaseParser | None:
        """
        Return the appropriate parser for a file.

        Returns None when the extension is unsupported.
        """
        extension = Path(file_path).suffix.lower()
        return self._parsers.get(extension)

    def supported_extensions(self) -> list[str]:
        """Return all extensions supported by registered parsers."""
        return list(self._parsers.keys())
