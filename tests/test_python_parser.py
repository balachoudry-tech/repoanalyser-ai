
"""
Tests for the Python AST parser.

These tests verify that the parser correctly identifies Python
functions, classes and asynchronous functions, and handles
source files containing invalid syntax.
"""

from src.parsers.python_parser import parse_python_code


def test_parse_python_function():
    """Verify extraction of a regular function."""

    source_code = """
def calculate_total(price, quantity):
    return price * quantity
"""

    result = parse_python_code(source_code, "main.py")

    assert len(result) == 1
    assert result[0]["symbol_name"] == "calculate_total"
    assert result[0]["symbol_type"] == "function"
    assert result[0]["file_path"] == "main.py"


def test_parse_python_class():
    """Verify extraction of a class definition."""

    source_code = """
class OrderService:
    def create_order(self):
        pass
"""

    result = parse_python_code(source_code, "orders.py")

    assert any(
        unit["symbol_name"] == "OrderService"
        and unit["symbol_type"] == "class"
        for unit in result
    )


def test_parse_async_function():
    """Verify extraction of an asynchronous function."""

    source_code = """
async def fetch_orders():
    return []
"""

    result = parse_python_code(source_code, "orders.py")

    assert any(
        unit["symbol_name"] == "fetch_orders"
        for unit in result
    )


def test_invalid_python_code():
    """Verify that invalid Python syntax is handled gracefully."""

    source_code = "def broken(:"

    result = parse_python_code(source_code, "broken.py")

    assert result == []
