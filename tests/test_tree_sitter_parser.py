
"""
Unit tests for the Tree-sitter multi-language parser.

These tests verify that supported definitions are extracted
correctly from JavaScript, TypeScript and Go source code.
"""

from src.parsers.tree_sitter_parser import parse_source_code


def test_javascript_function():
    """Verify JavaScript function extraction."""

    source = """
function calculateTotal(price, quantity) {
    return price * quantity;
}
"""

    result = parse_source_code(
        source,
        "orders.js",
        "javascript",
    )

    assert any(
        unit["symbol_name"] == "calculateTotal"
        and unit["symbol_type"] == "function"
        for unit in result
    )


def test_typescript_class():
    """Verify TypeScript class extraction."""

    source = """
class OrderService {
    createOrder() {
        return true;
    }
}
"""

    result = parse_source_code(
        source,
        "orders.ts",
        "typescript",
    )

    assert any(
        unit["symbol_name"] == "OrderService"
        and unit["symbol_type"] == "class"
        for unit in result
    )


def test_go_function():
    """Verify Go function extraction."""

    source = """
package main

func CalculateTotal(price int, quantity int) int {
    return price * quantity
}
"""

    result = parse_source_code(
        source,
        "orders.go",
        "go",
    )

    assert any(
        unit["symbol_name"] == "CalculateTotal"
        and unit["symbol_type"] == "function"
        for unit in result
    )


def test_unsupported_language():
    """Verify unsupported languages raise a clear error."""

    try:
        parse_source_code("some code", "file.rb", "ruby")
        assert False, "Expected ValueError"

    except ValueError as error:
        assert "Unsupported language" in str(error)
