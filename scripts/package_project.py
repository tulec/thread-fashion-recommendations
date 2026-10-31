"""Package source only; never bundle competition data, credentials or models."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(__file__).resolve().parents[1]
destination = root / "artifacts" / "thread-portfolio-source.zip"
destination.parent.mkdir(exist_ok=True)
files = [root / name for name in ["README.md", "app.py", "requirements.txt", "pyproject.toml", ".gitignore"]]
for directory in ["fashion_recs", "scripts", "tests", "docs", "notebooks", ".github", ".streamlit"]:
    files.extend(p for p in (root / directory).rglob("*") if p.is_file()
                 and "__pycache__" not in p.parts and p.suffix != ".pyc" and p.name != "secrets.toml")
with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
    for path in sorted(files):
        archive.write(path, path.relative_to(root).as_posix())
print(destination)

