from src.ingestion.repository_reader import scan_repository


files = scan_repository("./sample-repo")

print(f"Files discovered: {len(files)}")

for file in files:
    print("--------------------")
    print("Name:", file["file_name"])
    print("Path:", file["file_path"])
    print("Extension:", file["extension"])
    print("Content:")
    print(file["content"])