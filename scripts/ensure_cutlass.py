#!/usr/bin/env python3
"""Download and unpack CUTLASS v4.2.1 into third_party/cutlass."""

from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from pathlib import Path

CUTLASS_VERSION = "4.2.1"
CUTLASS_TAG = f"v{CUTLASS_VERSION}"
CUTLASS_URL = f"https://github.com/NVIDIA/cutlass/archive/refs/tags/{CUTLASS_TAG}.tar.gz"
CUTLASS_SHA256 = "a4513ba33ae82fd754843c6d8437bee1ac71a6ef1c74df886de2338e3917d4df"

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party"
ARCHIVE = THIRD_PARTY / f"cutlass-{CUTLASS_TAG}.tar.gz"
DEST = THIRD_PARTY / "cutlass"


def download() -> None:
    THIRD_PARTY.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists():
        print(f"Archive already present: {ARCHIVE}")
        return
    print(f"Downloading CUTLASS {CUTLASS_VERSION} from {CUTLASS_URL}")
    with urllib.request.urlopen(CUTLASS_URL) as response, open(ARCHIVE, "wb") as fh:
        fh.write(response.read())
    print(f"Saved archive to {ARCHIVE}")


def verify() -> None:
    sha256 = hashlib.sha256()
    with open(ARCHIVE, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha256.update(chunk)
    digest = sha256.hexdigest()
    if digest != CUTLASS_SHA256:
        raise RuntimeError(
            f"SHA256 mismatch for {ARCHIVE}\n"
            f" expected: {CUTLASS_SHA256}\n"
            f"   actual: {digest}"
        )
    print(f"Verified SHA256 checksum for {ARCHIVE}")


def extract() -> None:
    if DEST.exists():
        print(f"CUTLASS already extracted at {DEST}")
        return

    print(f"Extracting {ARCHIVE} to {DEST}")
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        members = tar.getmembers()
        top_level = {m.name.split("/", 1)[0] for m in members}
        tar.extractall(path=THIRD_PARTY)

    candidates = sorted((THIRD_PARTY / name for name in top_level), key=lambda p: p.name)
    extracted_root = None
    for candidate in candidates:
        if candidate.is_dir() and candidate.name.startswith("cutlass"):
            extracted_root = candidate
            break

    if extracted_root is None:
        raise RuntimeError("Unable to determine extracted CUTLASS root directory.")

    extracted_root.rename(DEST)
    print(f"Extraction complete: {DEST}")


def main() -> int:
    download()
    verify()
    extract()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

