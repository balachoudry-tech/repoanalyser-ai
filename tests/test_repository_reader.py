
from pathlib import Path

import pytest

from src.ingestion.repository_reader import scan_repository


def test_scan_python_files(tmp_path):
    (tmp_path / "main.py").write_text(
        "def hello():\n    return 'world'\n"
    )

    files = scan_repository(str(tmp_path))

    assert len(files) == 1
    assert files[0]["file_name"] == "main.py"
    assert files[0]["extension"] == ".py"


def test_ignore_node_modules(tmp_path):
    node_modules = tmp_path / "node_modules"
    node_modules.mkdir()

    (node_modules / "test.js").write_text(
        "function test() {}"
    )

    files = scan_repository(str(tmp_path))

    assert files == []


def test_ignore_large_files(tmp_path):
    large_file = tmp_path / "large.py"
    large_file.write_text("x" * 1_000_001)

    files = scan_repository(str(tmp_path))

    assert files == []


def test_invalid_repository_path():
    with pytest.raises(ValueError):
        scan_repository("/path/that/does/not/exist")


def test_ignore_binary_files(tmp_path):
    binary_file = tmp_path / "binary.py"
    binary_file.write_bytes(b"print('hello')\x00\x01")

    files = scan_repository(str(tmp_path))

    assert files == []
