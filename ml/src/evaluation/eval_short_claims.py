"""Does the LLM do better than the classifier on SHORT text - the register users actually paste?"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from calibration_analysis import ece  # noqa: E402
from eval_crosscorpus import best_f1_threshold  # noqa: E402
from eval_llm_baseline import PROMPTS, build_few_shot, classify  # noqa: E402
from llm_client import LLMClient, approx_tokens, estimate_cost, is_enabled, load_llm_config  # noqa: E402

TRIVIAL_F1 = 2 / 3


def _results_dir() -> Path:
    return Path(os.environ.get("RESULTS_DIR", "ml/results"))


def _artifacts_dir() -> Path:
    return Path(os.environ.get("ARTIFACTS_DIR", "ml/artifacts"))


def tfidf_scores(texts: list[str]) -> np.ndarray | None:
    path = _artifacts_dir() / "baseline_tfidf_logreg.joblib"
    if not path.exists():
        return None
    pipe = joblib.load(path)["pipeline"]
    idx = list(pipe.named_steps["clf"].classes_).index(1)
    return pipe.predict_proba(texts)[:, idx]


def summarise(name: str, y: np.ndarray, s: np.ndarray) -> dict:
    _, _, _, bf1 = best_f1_threshold(y, s)
    row = {"scorer": name, "n": len(y), "roc_auc": float(roc_auc_score(y, s)),
           "best_f1": float(bf1), "vs_trivial": float(bf1) - TRIVIAL_F1, "ece": float(ece(y, s)),
           "flagged_false": float((s[y == 1] > 0.6).mean()),
           "flagged_true": float((s[y == 0] > 0.6).mean())}
    print(f"  {name:<26} ROC-AUC {row['roc_auc']:.4f}   best F1 {row['best_f1']:.4f} "
          f"({row['vs_trivial']:+.4f} vs trivial)   ECE {row['ece']:.4f}   "
          f"flagged: {row['flagged_false']:.0%} of false / {row['flagged_true']:.0%} of real")
    return row


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Short-claim register test: LLM against the classifier.")
    p.add_argument("--claims", default="ml/results/current_claims.csv")
    p.add_argument("--headlines", default="data/processed/fakerecogna2.csv")
    p.add_argument("--arm", default="digit_few_shot")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    claims_path = Path(args.claims)
    if not claims_path.exists():
        print(f"[short] {claims_path} not found. Build it with eval_current_claims.py first.")
        return 1
    with open(claims_path, encoding="utf-8") as fh:
        false_claims = [r["claim"].strip() for r in csv.DictReader(fh) if r.get("claim", "").strip()]

    corpus = pd.read_csv(args.headlines).dropna(subset=["text", "label"])
    corpus["label"] = corpus["label"].astype(int)
    if "title" not in corpus.columns:
        print("[short] the headline corpus has no 'title' column; regenerate it with the current "
              "prepare_faketruebr.py.")
        return 1
    pool = corpus[(corpus["label"] == 0) & corpus["title"].notna()].copy()
    pool["wc"] = pool["title"].astype(str).str.split().str.len()
    pool = pool[(pool["wc"] >= 4) & (pool["wc"] <= 60)]
    real_titles = pool["title"].astype(str).sample(
        min(len(false_claims), len(pool)), random_state=args.seed
    ).tolist()

    texts = false_claims + real_titles
    y = np.array([1] * len(false_claims) + [0] * len(real_titles))
    print(f"[short] {len(false_claims)} false claims (2025-26) + {len(real_titles)} real headlines "
          f"(2020-22 proxy)\n[short] median length: "
          f"{sorted(len(t.split()) for t in texts)[len(texts) // 2]} words")

    cfg = load_llm_config()
    model = cfg.get("model")
    system, user_tpl = PROMPTS["digit_direct"]
    if args.arm == "digit_few_shot":
        block = build_few_shot(corpus, corpus.iloc[0:0], args.seed, digit_scale=True)
        user_tpl = f"Exemplos:\n\n{block}\n\n---\n\n{user_tpl}"
    overhead = approx_tokens(system) + approx_tokens(user_tpl.format(text=""))
    est_in = sum(approx_tokens(t) + overhead for t in texts)
    cost = estimate_cost(cfg, model, 1, est_in, len(texts) * 2)
    print(f"[short] {len(texts)} calls, ~{est_in:,} input tokens, estimated R${cost:.2f}")

    if args.dry_run:
        print(f"\n[dry-run] no calls made. Example item: {texts[0][:120]!r}")
        return 0

    print("\n" + "=" * 92)
    print("SHORT TEXT - the register users actually paste")
    print("=" * 92)
    rows = []
    tf = tfidf_scores(texts)
    if tf is not None:
        rows.append(summarise("TF-IDF (served)", y, tf))
    else:
        print("  TF-IDF artefact not found; skipping the classifier arm.")

    if not is_enabled():
        print("  MARITACA_API_KEY not set, so the LLM arm was skipped and no call was made.")
        return 0

    client = LLMClient(cfg=cfg, model=model, run_name=f"short_claims__{args.arm}__{model}")
    scores, errors = classify(texts, client, system, user_tpl, args.workers, use_logprobs=True)
    client.save_raw()
    print(f"\n  [{args.arm}] {client.report()}")
    if errors:
        print(f"  [{args.arm}] {len(errors)} failed; first: {errors[0][:140]}")
    ok = [i for i, s in enumerate(scores) if s is not None]
    if len(ok) < len(scores) * 0.9:
        print("  Too many failures to report metrics honestly; aborting the LLM arm.")
        client.close()
        return 1
    llm = np.array([scores[i] for i in ok])
    y_ok = y[ok]
    rows.append(summarise(f"LLM {args.arm}", y_ok, llm))

    if tf is not None and len(ok) == len(scores):
        ens = (rankdata(tf) / len(tf) + rankdata(llm) / len(llm)) / 2
        rows.append(summarise("rank-average ensemble", y, ens))
        print(f"\n  Spearman(TF-IDF, LLM) on short text: {spearmanr(tf, llm).statistic:.4f}")
        print("  On ARTICLES the same pair correlated 0.44 and the ensemble gained +0.048 ROC-AUC.")

    print("\n  Reference points from articles: TF-IDF ~0.80, LLM ~0.79, ensemble 0.838.")
    print("  A large LLM advantage HERE and not there would mean the two models should be used for")
    print("  different input lengths, which is the length-aware design recorded on 2026-08-24.")
    print("\n  CAVEATS, both load-bearing: the negatives are 2020-22 headlines while the positives")
    print("  are 2025-26 claims, so era is confounded with the label; and ClaimReview carries no")
    print("  'verdadeiro' ratings in Portuguese, so no better negative class exists.")

    client.close()
    out = _results_dir() / "short_claims_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
