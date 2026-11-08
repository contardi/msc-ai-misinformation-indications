"""Prepare the FakeTrueBR corpus (Chavarro et al., 2023) into data/processed/faketruebr.csv."""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

_TEXT_COLS = ["text", "texto", "noticia", "notícia", "news", "conteudo", "conteúdo", "content", "body"]
_LABEL_COLS = ["label", "classe", "class", "rotulo", "rótulo", "fake", "target", "y"]
_FAKE_VALUES = {"fake", "falsa", "falso", "1", "1.0", "true_fake", "fakenews", "fake_news"}
_TRUE_VALUES = {"true", "verdadeira", "verdadeiro", "real", "0", "0.0", "legit", "legitimate"}

_DATE_COLS = ["date", "data", "published", "publicacao", "publicação", "data_publicacao", "datetime"]
_TITLE_COLS = ["title", "titulo", "título", "headline", "manchete"]
_URL_COLS = ["url", "link", "endereco", "endereço", "source_url"]
_CATEGORY_COLS = ["category", "categoria", "topic", "tema", "assunto", "secao", "seção"]
_OPTIONAL = [("date", _DATE_COLS), ("title", _TITLE_COLS), ("url", _URL_COLS), ("category", _CATEGORY_COLS)]

_PT_MONTHS = {
    "janeiro": 1, "fevereiro": 2, "marco": 3, "março": 3, "abril": 4, "maio": 5, "junho": 6,
    "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
}
_PT_LONG_RE = re.compile(r"(\d{1,2})\s+(?:de\s+)?([a-zà-ÿ]{3,})\.?\s+(?:de\s+)?(\d{4})", re.IGNORECASE)
_NUMERIC_RE = re.compile(r"(\d{1,4})[/.\-](\d{1,2})[/.\-](\d{2,4})")
_PREFIX_RE = re.compile(r"^\s*(?:publicado|atualizado|postado)\s+(?:em|no dia)\s+", re.IGNORECASE)


def _parse_date(value) -> str | None:
    """Best-effort parse to an ISO date string, or None."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s or s in {"nan", "none", "-"}:
        return None
    s = _PREFIX_RE.sub("", s)

    m = _PT_LONG_RE.search(s)
    if m:
        day, month_name, year = m.group(1), m.group(2), m.group(3)
        month = _PT_MONTHS.get(month_name)
        if not month:
            month = next((v for k, v in _PT_MONTHS.items() if k.startswith(month_name[:3])), None)
        if month:
            try:
                return date(int(year), month, int(day)).isoformat()
            except ValueError:
                return None

    m = _NUMERIC_RE.search(s)
    if m:
        a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year, month, day = (a, b, c) if len(m.group(1)) == 4 else (c, b, a)
        if year < 100:
            year += 2000 if year < 50 else 1900
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            return None
    return None


def _data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "data"))


def _read_any_csv(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw))


def _load_source(args) -> pd.DataFrame:
    if args.csv:
        return pd.read_csv(args.csv)
    if args.hf_dataset:
        from datasets import load_dataset

        ds = load_dataset(args.hf_dataset, split=args.hf_split)
        return ds.to_pandas()
    if args.source_url:
        req = urllib.request.Request(args.source_url, headers={"User-Agent": "faketruebr/1.0"})
        with urllib.request.urlopen(req) as resp:  # noqa: S310 (user-supplied, trusted)
            raw = resp.read()
        if args.source_url.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                csv_name = next((n for n in zf.namelist() if n.lower().endswith(".csv")), None)
                if not csv_name:
                    raise ValueError("No .csv found inside the FakeTrueBR zip.")
                return _read_any_csv(zf.read(csv_name))
        return _read_any_csv(raw)
    raise ValueError(
        "No source given. Provide --csv, --source-url or --hf-dataset (see data/data-README.md). "
        "Training will use Fake.Br alone until FakeTrueBR is supplied."
    )


def _pick(colnames: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in colnames}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _to_label(value) -> int | None:
    s = str(value).strip().lower()
    if s in _FAKE_VALUES:
        return 1
    if s in _TRUE_VALUES:
        return 0
    return None


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    text_col = _pick(list(df.columns), _TEXT_COLS)
    label_col = _pick(list(df.columns), _LABEL_COLS)
    if not text_col or not label_col:
        raise ValueError(
            f"Could not auto-detect text/label columns in {list(df.columns)}. "
            "Rename them to 'text' and 'label' (1=fake, 0=true) and retry."
        )
    out = pd.DataFrame()
    out["text"] = (
        df[text_col].astype(str)
        .str.replace(r"\bnone\b", " ", case=False, regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    out["label"] = df[label_col].map(_to_label)

    for name, candidates in _OPTIONAL:
        col = _pick(list(df.columns), candidates)
        if not col:
            continue
        if name == "date":
            parsed = df[col].map(_parse_date)
            ok = int(parsed.notna().sum())
            print(f"[faketruebr] date column '{col}': {ok}/{len(parsed)} parsed "
                  f"({ok / max(1, len(parsed)):.1%})")
            out["date"] = parsed
        else:
            out[name] = df[col].astype(str).str.strip()

    out = out.dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    out = out[out["text"].str.len() > 0]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare FakeTrueBR into a labelled CSV.")
    parser.add_argument("--csv", default=None, help="Local CSV path.")
    parser.add_argument("--source-url", default=os.environ.get("FAKETRUEBR_URL", ""), help="CSV or ZIP URL.")
    parser.add_argument("--hf-dataset", default=os.environ.get("FAKETRUEBR_HF", ""), help="HuggingFace dataset id.")
    parser.add_argument("--hf-split", default="train", help="HuggingFace split (default: train).")
    parser.add_argument("--out", default=None, help="Output CSV (default: <DATA_DIR>/processed/faketruebr.csv).")
    parser.add_argument(
        "--invert-label",
        action="store_true",
        help="Flip labels after mapping (for datasets where 0=fake, 1=real, e.g. FakeRecogna 1.0).",
    )
    args = parser.parse_args(argv)
    args.source_url = args.source_url or None
    args.hf_dataset = args.hf_dataset or None

    out_path = Path(args.out) if args.out else _data_dir() / "processed" / "faketruebr.csv"

    try:
        df = normalise(_load_source(args))
    except Exception as exc:  # noqa: BLE001 - report clearly; do not fabricate data
        print(f"[faketruebr] ERROR: {exc}", file=sys.stderr)
        return 1

    if args.invert_label:
        df["label"] = 1 - df["label"]

    n_fake = int(df["label"].sum())
    print(f"[faketruebr] rows: {len(df)}  (fake={n_fake}, true={len(df) - n_fake})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[faketruebr] wrote -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
