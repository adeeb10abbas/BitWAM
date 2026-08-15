#!/usr/bin/env python3
"""Download and verify a public Google Cloud Storage prefix without gsutil."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GCSObject:
    name: str
    size: int
    md5: str | None


def list_objects(bucket: str, prefix: str) -> list[GCSObject]:
    """Return every non-empty object below a public GCS prefix."""
    objects: list[GCSObject] = []
    page_token: str | None = None
    while True:
        query = {"prefix": prefix, "maxResults": "1000"}
        if page_token:
            query["pageToken"] = page_token
        url = (
            f"https://storage.googleapis.com/storage/v1/b/{urllib.parse.quote(bucket)}/o?"
            + urllib.parse.urlencode(query)
        )
        with urllib.request.urlopen(url, timeout=60) as response:
            page = json.load(response)
        for item in page.get("items", []):
            size = int(item.get("size", 0))
            if size:
                objects.append(
                    GCSObject(name=item["name"], size=size, md5=item.get("md5Hash"))
                )
        page_token = page.get("nextPageToken")
        if not page_token:
            return objects


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return base64.b64encode(digest.digest()).decode()


def _target_path(obj: GCSObject, source_prefix: str, output_dir: Path) -> Path:
    relative = obj.name.removeprefix(source_prefix).lstrip("/")
    if not relative or relative.startswith("../"):
        raise ValueError(f"Object is outside requested prefix: {obj.name}")
    return output_dir / relative


def download_object(
    bucket: str,
    obj: GCSObject,
    source_prefix: str,
    output_dir: Path,
    retries: int,
) -> tuple[str, int]:
    """Download one object atomically, resuming a partial file when possible."""
    target = _target_path(obj, source_prefix, output_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    if (
        target.is_file()
        and target.stat().st_size == obj.size
        and (obj.md5 is None or _md5(target) == obj.md5)
    ):
        return "skipped", obj.size

    partial = target.with_name(f".{target.name}.part")
    if partial.is_file() and partial.stat().st_size == obj.size:
        if obj.md5 is None or _md5(partial) == obj.md5:
            os.replace(partial, target)
            return "downloaded", obj.size
        partial.unlink()
    encoded_name = urllib.parse.quote(obj.name, safe="")
    url = (
        f"https://storage.googleapis.com/download/storage/v1/b/{urllib.parse.quote(bucket)}"
        f"/o/{encoded_name}?alt=media"
    )
    for attempt in range(retries + 1):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > obj.size:
                partial.unlink()
                offset = 0
            request = urllib.request.Request(url)
            if offset:
                request.add_header("Range", f"bytes={offset}-")
            with urllib.request.urlopen(request, timeout=120) as response:
                mode = "ab" if offset and response.status == 206 else "wb"
                with partial.open(mode) as stream:
                    while block := response.read(8 * 1024 * 1024):
                        stream.write(block)
            if partial.stat().st_size != obj.size:
                raise OSError(
                    f"size mismatch for {obj.name}: {partial.stat().st_size} != {obj.size}"
                )
            if obj.md5 is not None and _md5(partial) != obj.md5:
                partial.unlink()
                raise OSError(f"MD5 mismatch for {obj.name}")
            os.replace(partial, target)
            return "downloaded", obj.size
        except Exception:
            if attempt == retries:
                raise
            time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default="gresearch")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--manifest", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.workers < 1 or args.retries < 0:
        raise SystemExit("workers must be positive and retries must be non-negative")
    started = time.time()
    objects = list_objects(args.bucket, args.prefix)
    if not objects:
        raise SystemExit(f"No objects found: gs://{args.bucket}/{args.prefix}")
    expected_bytes = sum(obj.size for obj in objects)
    print(f"Found {len(objects)} objects ({expected_bytes / 2**30:.2f} GiB)", flush=True)
    counts = {"downloaded": 0, "skipped": 0}
    failures: list[dict[str, str]] = []
    completed_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_object,
                args.bucket,
                obj,
                args.prefix,
                args.output_dir,
                args.retries,
            ): obj
            for obj in objects
        }
        for index, future in enumerate(as_completed(futures), 1):
            try:
                status, size = future.result()
            except Exception as error:
                obj = futures[future]
                failures.append({"object": obj.name, "error": repr(error)})
                print(
                    f"[{index}/{len(objects)}] FAILED {obj.name}: {error!r}",
                    flush=True,
                )
                continue
            counts[status] += 1
            completed_bytes += size
            print(
                f"[{index}/{len(objects)}] {completed_bytes / 2**30:.2f}/"
                f"{expected_bytes / 2**30:.2f} GiB {futures[future].name}",
                flush=True,
            )

    if failures:
        print(json.dumps({"verified": False, "failures": failures}, indent=2), flush=True)
        return 1

    manifest = args.manifest or args.output_dir / "download_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source": f"gs://{args.bucket}/{args.prefix}",
        "output_dir": str(args.output_dir.resolve()),
        "objects": len(objects),
        "bytes": expected_bytes,
        **counts,
        "elapsed_seconds": time.time() - started,
        "verified": True,
    }
    temporary = manifest.with_suffix(f"{manifest.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest)
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
