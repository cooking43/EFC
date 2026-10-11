"""Verify the byte-level integrity of the published revision (no dependencies)."""
from pathlib import Path
import gzip
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def digest(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        result.update(block)
    return result.hexdigest()


def main():
    manifest = json.loads((ROOT / "release_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        target = ROOT / item["path"]
        with target.open("rb") as stream:
            assert digest(stream) == item["sha256"], item["path"]
        assert target.stat().st_size == item["bytes"], item["path"]
        if "uncompressed_sha256" in item:
            with gzip.open(target, "rb") as stream:
                assert digest(stream) == item["uncompressed_sha256"], item["path"]
    index = json.loads((ROOT / "model_index.json").read_text(encoding="utf-8"))
    for model in index:
        with (ROOT / model["checkpoint"]).open("rb") as stream:
            assert digest(stream) == model["checkpoint_sha256"], model["checkpoint"]
    print(f"Verified {len(manifest['files'])} files, {len(index)} frozen models, and lossless prediction CSV compression.")


if __name__ == "__main__":
    main()
