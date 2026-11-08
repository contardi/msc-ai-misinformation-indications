"""Shared pieces of every blind coding sheet in this project."""

from __future__ import annotations

import pandas as pd

MAX_CHARS = 6000


def assert_alignment(sample: pd.DataFrame, saved: pd.DataFrame) -> None:
    """Refuse to continue unless the re-derived rows ARE the rows that were scored."""
    if len(sample) != len(saved):
        raise SystemExit(f"[sheet] row count differs: corpus {len(sample)} vs saved {len(saved)}")
    label_match = (sample["label"].to_numpy() == saved["label"].to_numpy()).mean()
    if label_match < 1.0:
        raise SystemExit(f"[sheet] labels do not align ({label_match:.4f}) - the sample has drifted")
    if "category" in saved.columns:
        cat_match = (
            sample["category"].fillna("").to_numpy() == saved["category"].fillna("").to_numpy()
        ).mean()
        if cat_match < 1.0:
            raise SystemExit(f"[sheet] categories do not align ({cat_match:.4f}) - sample drifted")
    print(f"[sheet] alignment verified: {len(sample)} rows, labels and categories match exactly")


def body(text: str, limit: int = MAX_CHARS) -> str:
    """The article as the coder reads it, which is the article as the MODEL reads it."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."
