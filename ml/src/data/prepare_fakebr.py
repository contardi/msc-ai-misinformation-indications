"""Download and prepare the Fake.Br corpus into a single labelled CSV."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

FAKEBR_ZIP_URL = "https://codeload.github.com/roneysco/Fake.br-Corpus/zip/refs/heads/master"

TRUE_DIRNAME = "true"
FAKE_DIRNAME = "fake"
PREFERRED_PARENT = "size_normalized_texts"


def _data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "data"))


def download(url: str, dest: Path) -> Path:
    """Download ``url`` to ``dest`` unless it already exists (idempotent)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[prepare] using cached archive: {dest}")
        return dest
    print(f"[prepare] downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "fakebr-prepare/1.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as fh:  # noqa: S310 (trusted URL)
        fh.write(resp.read())
    print(f"[prepare] saved archive: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def extract(zip_path: Path, dest_dir: Path) -> Path:
    """Extract ``zip_path`` into ``dest_dir`` and return the extraction root."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    print(f"[prepare] extracted to: {dest_dir}")
    return dest_dir


def _read_text(path: Path) -> str:
    """Read a corpus text file, tolerating the occasional non-UTF-8 article."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1").strip()


def _find_label_dir(root: Path, label_name: str, preferred_parent: str) -> Path | None:
    """Locate the directory of articles for one label."""
    candidates = [d for d in root.rglob(label_name) if d.is_dir()]
    candidates = [d for d in candidates if any(d.glob("*.txt"))]
    if not candidates:
        return None
    preferred = [d for d in candidates if d.parent.name == preferred_parent]
    return (preferred or candidates)[0]


def parse_corpus(root: Path, preferred_parent: str = PREFERRED_PARENT) -> list[tuple[str, int]]:
    """Walk the extracted corpus and return ``(text, label)`` rows."""
    true_dir = _find_label_dir(root, TRUE_DIRNAME, preferred_parent)
    fake_dir = _find_label_dir(root, FAKE_DIRNAME, preferred_parent)
    if true_dir is None or fake_dir is None:
        raise FileNotFoundError(
            f"Could not locate '{TRUE_DIRNAME}'/'{FAKE_DIRNAME}' text folders under {root}. "
            "The archive layout may have changed - pass a different source with --source-url."
        )
    print(f"[prepare] true articles: {true_dir}")
    print(f"[prepare] fake articles: {fake_dir}")

    rows: list[tuple[str, int]] = []
    for label, directory in ((0, true_dir), (1, fake_dir)):
        for txt in sorted(directory.glob("*.txt")):
            body = _read_text(txt)
            if body:
                rows.append((body, label))
    return rows


def write_csv(rows: list[tuple[str, int]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    print(f"[prepare] wrote {len(rows)} rows -> {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the Fake.Br corpus into a labelled CSV.")
    parser.add_argument("--source-url", default=FAKEBR_ZIP_URL, help="Fake.Br archive (zip) URL.")
    parser.add_argument(
        "--fallback-url",
        default=os.environ.get("FAKEBR_FALLBACK_URL", ""),
        help="FakeTrueBR (or other) archive zip to use if the primary source is unreachable.",
    )
    parser.add_argument("--out", default=None, help="Output CSV path (default: <DATA_DIR>/processed/fakebr.csv).")
    parser.add_argument("--cache-dir", default=None, help="Where to cache the raw archive (default: <DATA_DIR>/raw).")
    parser.add_argument(
        "--version",
        default=PREFERRED_PARENT,
        choices=["size_normalized_texts", "full_texts"],
        help="Which Fake.Br text variant to parse (default: size_normalized_texts).",
    )
    args = parser.parse_args(argv)

    data_dir = _data_dir()
    out_path = Path(args.out) if args.out else data_dir / "processed" / "fakebr.csv"
    cache_dir = Path(args.cache_dir) if args.cache_dir else data_dir / "raw"

    sources = [("Fake.Br", args.source_url)]
    if args.fallback_url:
        sources.append(("FakeTrueBR (fallback)", args.fallback_url))

    last_error: Exception | None = None
    for name, url in sources:
        try:
            zip_path = download(url, cache_dir / "corpus.zip")
            extract_root = extract(zip_path, cache_dir / "corpus")
            rows = parse_corpus(extract_root, args.version)
            n_fake = sum(label for _, label in rows)
            n_true = len(rows) - n_fake
            print(f"[prepare] corpus used: {name}")
            print(f"[prepare] class balance - true(0): {n_true}  fake(1): {n_fake}")
            write_csv(rows, out_path)
            return 0
        except Exception as exc:  # noqa: BLE001 - report and try the next source
            last_error = exc
            print(f"[prepare] source '{name}' failed: {exc}", file=sys.stderr)
            corpus_zip = cache_dir / "corpus.zip"
            if corpus_zip.exists():
                corpus_zip.unlink()

    print(
        "[prepare] ERROR: no corpus could be prepared. Provide a working --source-url or "
        "--fallback-url (a Fake.Br or FakeTrueBR archive zip).",
        file=sys.stderr,
    )
    if last_error:
        print(f"[prepare] last error: {last_error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
