
from pathlib import Path

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go",
    ".java", ".php", ".cs", ".rb", ".rs",
    ".cpp", ".c", ".h", ".html", ".css",
    ".sql", ".json", ".yaml", ".yml", ".toml", ".md",
}

IGNORED_DIRECTORIES = {
    ".git", ".venv", "venv", "node_modules",
    "__pycache__", "dist", "build",
    ".idea", ".vscode", ".pytest_cache",
}

MAX_FILE_SIZE_BYTES = 1_000_000
MAX_FILES = 10_000


def should_ignore(path: Path) -> bool:
    return any(part in IGNORED_DIRECTORIES for part in path.parts)


def is_binary_file(file_path: Path) -> bool:
    try:
        with file_path.open("rb") as file:
            return b"\0" in file.read(8192)
    except OSError:
        return True


def scan_repository(repository_path: str) -> list[dict]:
    root = Path(repository_path).expanduser().resolve()

    if not root.exists():
        raise ValueError("Repository path does not exist")

    if not root.is_dir():
        raise ValueError("Repository path must be a directory")

    files = []

    for file_path in root.rglob("*"):

        if file_path.is_symlink() or not file_path.is_file():
            continue

        relative_path = file_path.relative_to(root)

        if should_ignore(relative_path):
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            size = file_path.stat().st_size

            if size > MAX_FILE_SIZE_BYTES:
                continue

            if is_binary_file(file_path):
                continue

            content = file_path.read_text(
                encoding="utf-8",
                errors="strict",
            )

            files.append({
                "file_name": file_path.name,
                "file_path": relative_path.as_posix(),
                "extension": file_path.suffix.lower(),
                "content": content,
                "size": size,
            })

            if len(files) >= MAX_FILES:
                break

        except (OSError, UnicodeDecodeError):
            continue

    return files
