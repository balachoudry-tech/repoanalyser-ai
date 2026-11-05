
from src.ingestion.ingestion_service import IngestionService


def test_ingest_python_repository(tmp_path):
    source_file = tmp_path / "main.py"

    source_file.write_text(
        """
def hello():
    return "world"


class UserService:
    pass
"""
    )

    service = IngestionService()

    result = service.ingest_repository(str(tmp_path))

    assert result["total_files"] == 1
    assert result["parsed_files"] == 1
    assert result["total_code_units"] == 2
    assert result["failed_files"] == 0

    symbols = [
        unit["symbol_name"]
        for unit in result["code_units"]
    ]

    assert "hello" in symbols
    assert "UserService" in symbols


def test_ingest_unsupported_files(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("database: postgres")

    service = IngestionService()

    result = service.ingest_repository(str(tmp_path))

    assert result["total_files"] == 1
    assert result["unsupported_files"] == 1
    assert result["parsed_files"] == 0


def test_ingest_empty_repository(tmp_path):
    service = IngestionService()

    result = service.ingest_repository(str(tmp_path))

    assert result["total_files"] == 0
    assert result["parsed_files"] == 0
    assert result["total_code_units"] == 0


def test_ingest_multiple_languages(tmp_path):
    (tmp_path / "main.py").write_text(
        "def python_function():\n    pass"
    )

    (tmp_path / "app.js").write_text(
        "function javascriptFunction() { return true; }"
    )

    service = IngestionService()

    result = service.ingest_repository(str(tmp_path))

    assert result["total_files"] == 2
    assert result["parsed_files"] == 2
    assert result["total_code_units"] >= 2
