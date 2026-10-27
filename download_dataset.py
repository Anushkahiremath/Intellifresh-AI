"""Download the public Freshness44 v1 archive and safely extract image files."""
import argparse
import json
import time
import zipfile
from pathlib import Path

import requests

URL = "https://www.kaggle.com/api/v1/datasets/download/siavash93/freshness44?datasetVersionNumber=1"


def download_archive(root: Path, force: bool = False) -> Path:
    archive = root / "freshness44.zip"
    partial = root / "freshness44.zip.part"

    if archive.exists() and archive.stat().st_size > 0 and not force:
        print(f"Using cached archive at {archive}")
        return archive

    if partial.exists() and partial.stat().st_size > 0 and not force:
        print(f"Resuming partial download from {partial}")
        start = partial.stat().st_size
    else:
        partial.unlink(missing_ok=True)
        start = 0

    headers = {"Range": f"bytes={start}-"} if start else {}
    with requests.get(URL, stream=True, timeout=(30, 120), headers=headers) as response:
        response.raise_for_status()
        expected = int(response.headers.get("Content-Length", 0))
        downloaded = start
        last = time.monotonic()
        with partial.open("ab" if start else "wb") as file:
            for chunk in response.iter_content(4 * 1024 * 1024):
                if not chunk:
                    continue
                file.write(chunk)
                downloaded += len(chunk)
                if time.monotonic() - last > 20:
                    print(f"Downloaded {downloaded / 1e9:.2f} GB / {expected / 1e9:.2f} GB", flush=True)
                    last = time.monotonic()
        if expected and downloaded != start + expected:
            raise RuntimeError("Incomplete archive download")
    partial.replace(archive)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download the archive even if a cached copy exists")
    args = parser.parse_args()

    root = Path("raw_data/freshness44").resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = download_archive(root, force=args.force)
    count = 0
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            if Path(member.filename).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                continue
            destination = (root / member.filename).resolve()
            if root not in destination.parents:
                raise ValueError("Unsafe archive path")
            if not destination.exists() or destination.stat().st_size != member.file_size:
                zipped.extract(member, root)
            count += 1
    metadata = {"source": "https://www.kaggle.com/datasets/siavash93/freshness44", "version": 1,
                "archive_bytes": archive.stat().st_size, "extracted_images": count,
                "license": "Unknown in Kaggle metadata; preserve provenance and verify before redistribution",
                "scope": "Image freshness labels only; no measured environmental or shelf-life targets"}
    (root / "source.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
