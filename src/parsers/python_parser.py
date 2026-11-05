
"""
Python source code parser.

Uses Python's built-in AST module to identify classes and functions.
Extracted definitions are returned in a normalized format for
downstream indexing and retrieval.
"""

import ast

from src.parsers.base import BaseParser


class PythonParser(BaseParser):
    """Parse Python source files and extract code definitions."""

    def parse(self, source_code: str, file_path: str) -> list[dict]:
        """
        Extract classes and functions from Python source code.

        Args:
            source_code: Complete Python source file contents.
            file_path: Repository-relative source file path.

        Returns:
            A list of normalized code-unit dictionaries.
            Returns an empty list when the source has invalid syntax.
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

        code_units = []

        for node in ast.walk(tree):

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                code_units.append({
                    "symbol_name": node.name,
                    "symbol_type": "function",
                    "file_path": file_path,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "content": ast.get_source_segment(source_code, node),
                    "language": "python",
                })

            elif isinstance(node, ast.ClassDef):
                code_units.append({
                    "symbol_name": node.name,
                    "symbol_type": "class",
                    "file_path": file_path,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "content": ast.get_source_segment(source_code, node),
                    "language": "python",
                })

        return code_units
def parse_python_code(source_code: str, file_path: str = "<unknown>") -> list[dict]:
    """
    Convenience function to parse Python source code.

    Maintains compatibility with function-based consumers.
    """
    parser = PythonParser()
    return parser.parse(source_code, file_path)