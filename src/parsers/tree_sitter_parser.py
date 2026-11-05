
from tree_sitter_language_pack import get_parser

from src.parsers.base import BaseParser


LANGUAGE_CONFIG = {
    "javascript": {
        "function": {
            "function_declaration",
            "method_definition",
            "arrow_function",
            "function_expression",
        },
        "class": {"class_declaration"},
    },
    "typescript": {
        "function": {
            "function_declaration",
            "method_definition",
            "arrow_function",
            "function_expression",
        },
        "class": {"class_declaration"},
    },
    "go": {
        "function": {
            "function_declaration",
            "method_declaration",
        },
        "class": set(),
    },
}


class TreeSitterParser(BaseParser):
    """Multi-language parser using Tree-sitter."""

    def __init__(self, language: str):
        if language not in LANGUAGE_CONFIG:
            raise ValueError(f"Unsupported language: {language}")

        self.language = language
        self.parser = get_parser(language)

    def parse(self, source_code: str, file_path: str) -> list[dict]:
        source_bytes = source_code.encode("utf-8")
        tree = self.parser.parse(source_bytes)

        definitions = LANGUAGE_CONFIG[self.language]
        code_units = []

        def visit(node, parent_symbol=None):

            current_symbol = parent_symbol

            for symbol_type, node_types in definitions.items():

                if node.type in node_types:
                    name_node = node.child_by_field_name("name")

                    symbol_name = (
                        source_bytes[
                            name_node.start_byte:name_node.end_byte
                        ].decode("utf-8")
                        if name_node
                        else "<anonymous>"
                    )

                    code_units.append({
                        "symbol_name": symbol_name,
                        "symbol_type": symbol_type,
                        "parent_symbol": parent_symbol,
                        "file_path": file_path,
                        "start_line": node.start_point.row + 1,
                        "end_line": node.end_point.row + 1,
                        "content": source_bytes[
                            node.start_byte:node.end_byte
                        ].decode("utf-8"),
                        "language": self.language,
                    })

                    current_symbol = symbol_name
                    break

            for child in node.children:
                visit(child, current_symbol)

        visit(tree.root_node)

        return code_units

def parse_source_code(
    source_code: str,
    file_path: str,
    language: str,
) -> list[dict]:
    """
    Backward-compatible function interface.
    """
    parser = TreeSitterParser(language)

    return parser.parse(
        source_code=source_code,
        file_path=file_path,
    )
