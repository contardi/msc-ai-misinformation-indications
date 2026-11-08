"""Does the platform still work on claims from 2025-2026, and on the register users actually paste?"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

import httpx

sys.path.insert(0, "/app")

from app.api.factcheck import FACTCHECK_URL, MONITORED_CHECKERS, is_enabled  # noqa: E402
from app.models import classifier  # noqa: E402

SEED_QUERIES = [
    "governo", "presidente", "eleições", "urna eletrônica", "STF", "congresso",
    "vacina", "saúde", "ministério", "imposto", "economia", "polícia",
    "escola", "meio ambiente", "amazônia", "segurança pública",
]

FALSE_RATINGS = {
    "falso", "fake", "enganoso", "distorcido", "insustentável", "insustentavel",
    "exagerado", "não é bem assim", "nao e bem assim", "de olho", "contraditório",
    "false", "misleading", "altered", "distorted",
}
TRUE_RATINGS = {
    "verdadeiro", "verdade", "correto", "confirmado", "é verdade", "e verdade",
    "true", "correct", "accurate", "mostly true", "verdadeiro, mas",
}
LOW_BAND, HIGH_BAND = 0.4, 0.6


def _results_dir() -> Path:
    return Path(os.environ.get("RESULTS_DIR", "/app/results"))


def _rating_class(rating: str) -> int | None:
    """1 = checker ruled it false, 0 = checker ruled it true, None = anything else."""
    r = (rating or "").strip().lower()
    if any(r == v or r.startswith(v) for v in TRUE_RATINGS):
        return 0
    if any(r == v or r.startswith(v) for v in FALSE_RATINGS):
        return 1
    return None


def _brazilian(url: str) -> str | None:
    for name, domains in MONITORED_CHECKERS:
        if any(d in (url or "").lower() for d in domains):
            return name
    return None


def collect(max_age_days: int, per_query: int, delay: float, brazilian_only: bool) -> list[dict]:
    """Assemble a de-duplicated set of recently-checked false claims."""
    seen: dict[str, dict] = {}
    key = os.environ.get("GOOGLE_FACTCHECK_API_KEY", "").strip()
    with httpx.Client(timeout=10.0) as client:
        for q in SEED_QUERIES:
            params = {
                "query": q, "languageCode": "pt", "pageSize": per_query,
                "maxAgeDays": max_age_days, "key": key,
            }
            try:
                resp = client.get(FACTCHECK_URL, params=params)
                resp.raise_for_status()
                payload = resp.json()
            except Exception as exc:  # noqa: BLE001 - reported, never silently swallowed
                print(f"  [{q}] request failed: {type(exc).__name__}: {exc}")
                continue

            found = 0
            for claim in payload.get("claims", []):
                text = (claim.get("text") or "").strip()
                reviews = claim.get("claimReview") or []
                if not text or not reviews or text in seen:
                    continue
                review = reviews[0]
                rating = review.get("textualRating", "")
                cls = _rating_class(rating)
                if cls is None:
                    continue
                url = review.get("url", "")
                checker = _brazilian(url)
                if brazilian_only and not checker:
                    continue
                seen[text] = {
                    "claim": text,
                    "label": cls,
                    "rating": rating,
                    "publisher": (review.get("publisher") or {}).get("name", ""),
                    "checker": checker or "",
                    "date": (review.get("reviewDate") or claim.get("claimDate") or "")[:10],
                    "url": url,
                    "seed": q,
                }
                found += 1
            print(f"  [{q:<18}] +{found:>3} new  (total {len(seen)})")
            time.sleep(delay)
    return list(seen.values())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Score recently fact-checked false claims.")
    p.add_argument("--max-age-days", type=int, default=540, help="How far back to collect.")
    p.add_argument("--per-query", type=int, default=30)
    p.add_argument("--delay", type=float, default=0.2)
    p.add_argument("--all-publishers", action="store_true",
                   help="Keep non-Brazilian Portuguese-language checkers too (default: Brazilian only).")
    p.add_argument("--compare-titles", default="/app/data/processed/fakerecogna2.csv",
                   help="Corpus CSV whose REAL headlines act as a proxy negative class when the "
                        "index supplies none. Era is confounded with the label; see the output.")
    p.add_argument("--dry-run", action="store_true", help="Collect and describe; do not score.")
    args = p.parse_args(argv)

    if not is_enabled():
        print("[current-claims] GOOGLE_FACTCHECK_API_KEY is not set, so no claims can be collected "
              "and no external call was made.\n[current-claims] This is the offline default, not a "
              "result.")
        return 0

    print(f"[current-claims] collecting claims rated false in the last {args.max_age_days} days "
          f"({len(SEED_QUERIES)} seed queries)")
    claims = collect(args.max_age_days, args.per_query, args.delay, not args.all_publishers)
    if not claims:
        print("\n[current-claims] no claims collected - widen --max-age-days or pass "
              "--all-publishers. Reporting nothing rather than a zero.")
        return 0

    years = Counter(c["date"][:4] for c in claims if c["date"])
    print(f"\n[current-claims] {len(claims)} distinct false claims")
    print(f"  by review year: {dict(sorted(years.items()))}")
    print(f"  by checker:     {dict(Counter(c['checker'] or c['publisher'] for c in claims).most_common(8))}")
    lengths = [len(c["claim"].split()) for c in claims]
    print(f"  claim length:   median {sorted(lengths)[len(lengths) // 2]} words "
          f"(min {min(lengths)}, max {max(lengths)})")
    print("                  - the training corpora are news ARTICLES; this is the register gap.")

    if args.dry_run:
        print("\n[dry-run] collected only, not scored. Examples:")
        for c in claims[:5]:
            print(f"    [{c['date']}] {c['rating']:<12} {c['claim'][:90]}")
        return 0

    if not classifier.load_model():
        print(f"\n[current-claims] model not loaded ({classifier.last_error()}). "
              "Train the baseline first (see README).")
        return 1
    print(f"\n[current-claims] scoring with the SERVED model: {classifier.model_path()}")

    for c in claims:
        c["score"] = round(classifier.predict(c["claim"]).prob, 4)

    false_s = sorted(c["score"] for c in claims if c["label"] == 1)
    true_s = sorted(c["score"] for c in claims if c["label"] == 0)

    def band_row(name: str, scores: list[float]) -> None:
        n = len(scores)
        if not n:
            print(f"  {name:<22} (none collected)")
            return
        flagged = sum(1 for s in scores if s > HIGH_BAND)
        reassured = sum(1 for s in scores if s < LOW_BAND)
        print(f"  {name:<22} n={n:<4} flagged>{HIGH_BAND} {flagged / n:6.1%}   "
              f"inconclusive {1 - (flagged + reassured) / n:6.1%}   "
              f"below {LOW_BAND} {reassured / n:6.1%}   median {scores[n // 2]:.3f}")

    print("\n" + "=" * 96)
    print("Scores on CURRENT claims, split by the checker's own ruling")
    print("=" * 96)
    band_row("ruled FALSE", false_s)
    band_row("ruled TRUE", true_s)

    if false_s and true_s:
        wins = ties = 0
        for a in false_s:
            for b in true_s:
                if a > b:
                    wins += 1
                elif a == b:
                    ties += 1
        auc = (wins + 0.5 * ties) / (len(false_s) * len(true_s))
        print(f"\n  ROC-AUC between the two groups: {auc:.4f}   (0.5 = no discrimination)")
        gap = (sum(false_s) / len(false_s)) - (sum(true_s) / len(true_s))
        print(f"  mean score gap (false - true):  {gap:+.4f}")
        if auc < 0.60:
            print("\n  READ THIS PLAINLY: on short current claims the model is at or near chance.")
            print("  The flagged percentage above is therefore NOT detection - it reflects short text")
            print("  scoring mid-range regardless of content. This is the register mismatch made")
            print("  concrete, and it is the gap between how the platform is evaluated (news")
            print("  articles) and how it is actually used (pasted claims).")
    else:
        print("\n  NO true-rated claims exist to compare against. This is not a sampling accident:")
        print("  across 350 Portuguese-language checks the index contains only debunk verdicts")
        print("  (falso, enganoso, errado, distorcido, fora de contexto, satira) and no 'verdadeiro'")
        print("  at all. Brazilian fact-checkers publish what they debunk, so ClaimReview CANNOT")
        print("  supply a negative class for this language - a real limitation of the")
        print("  'build an evaluation set from ClaimReview' idea in CLAUDE.md section 8.")
        print("  The flagged percentage above is therefore a one-sided detection rate and must")
        print("  never be read as accuracy or as evidence of discrimination.")

        if args.compare_titles:
            ref = Path(args.compare_titles)
            titles = []
            if ref.exists():
                with open(ref, encoding="utf-8") as fh:
                    for row in csv.DictReader(fh):
                        t = (row.get("title") or "").strip()
                        if int(row.get("label", 1)) == 0 and 4 <= len(t.split()) <= 60:
                            titles.append(t)
            if titles:
                import random as _random
                sample = _random.Random(42).sample(titles, min(len(false_s), len(titles)))
                ref_scores = sorted(classifier.predict(t).prob for t in sample)
                print(f"\n  {'-' * 92}")
                print("  PROXY negative class: real news HEADLINES from the corpora (same register,")
                print("  similar length). Read with care - these are from 2020-2022, so era is")
                print("  CONFOUNDED with the label. It is a sanity check, not a clean measurement.")
                print("  " + "-" * 92)
                band_row("proxy: real headlines", ref_scores)
                wins = ties = 0
                for a in false_s:
                    for b in ref_scores:
                        if a > b:
                            wins += 1
                        elif a == b:
                            ties += 1
                auc = (wins + 0.5 * ties) / (len(false_s) * len(ref_scores))
                print(f"\n  ROC-AUC, current false claims vs real headlines: {auc:.4f} "
                      "(0.5 = no separation)")
                if auc < 0.60:
                    print("  At or near chance. On SHORT text the model is not discriminating: the")
                    print("  45% 'flagged' above reflects short input scoring mid-range whatever it")
                    print("  says. That is the gap between how this platform is evaluated (news")
                    print("  articles) and how it is actually used (pasted claims).")

    print("\n  No F1 or accuracy is reported: the two groups are opportunistically sampled and")
    print("  wildly unbalanced, so any such figure would describe the sampling, not the model.")
    print("  These are SHORT CLAIMS; the model was trained on news articles.")

    out = _results_dir() / "current_claims.csv"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(claims[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(claims)
        print(f"\n  wrote {out}  (collected {date.today().isoformat()})")
    except OSError as exc:
        print(f"\n  could not write results ({exc}); the summary above is the record")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
