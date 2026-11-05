
from src.parsers.parser_registry import ParserRegistry
from src.parsers.python_parser import PythonParser
from src.parsers.tree_sitter_parser import TreeSitterParser


def test_python_parser_resolution():
    registry = ParserRegistry()

    parser = registry.get_parser("main.py")

    assert isinstance(parser, PythonParser)


def test_javascript_parser_resolution():
    registry = ParserRegistry()

    parser = registry.get_parser("app.js")

    assert isinstance(parser, TreeSitterParser)
    assert parser.language == "javascript"


def test_typescript_parser_resolution():
    registry = ParserRegistry()

    parser = registry.get_parser("app.ts")

    assert isinstance(parser, TreeSitterParser)
    assert parser.language == "typescript"


def test_go_parser_resolution():
    registry = ParserRegistry()

    parser = registry.get_parser("main.go")

    assert isinstance(parser, TreeSitterParser)
    assert parser.language == "go"


def test_unsupported_extension():
    registry = ParserRegistry()

    parser = registry.get_parser("config.yaml")

    assert parser is None


def test_supported_extensions():
    registry = ParserRegistry()

    extensions = registry.supported_extensions()

    assert ".py" in extensions
    assert ".js" in extensions
    assert ".ts" in extensions
    assert ".go" in extensions
