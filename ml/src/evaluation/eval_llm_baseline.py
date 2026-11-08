"""LLM as a zero/few-shot classification baseline against the fine-tuned encoder."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from calibration_analysis import ece  # noqa: E402
from eval_crosscorpus import best_f1_threshold, stratified_sample  # noqa: E402
from llm_client import (  # noqa: E402
    LLMClient,
    approx_tokens,
    estimate_cost,
    expected_digit,
    is_enabled,
    is_retryable,
    load_llm_config,
    retry_delay,
)

from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

TRIVIAL_F1 = 2 / 3
MAX_CHARS = 4000

_FORMAT = (
    "Responda APENAS com um número inteiro de 0 a 100, sem texto, sem símbolos e sem explicação. "
    "0 = certamente informação legítima. 100 = certamente desinformação."
)

PROMPTS: dict[str, tuple[str, str]] = {
    "zero_direct": (
        "Você avalia notícias em português brasileiro. " + _FORMAT,
        "Qual a probabilidade (0-100) de este texto conter desinformação?\n\nTexto:\n{text}",
    ),
    "zero_analytic": (
        "Você é um verificador de fatos brasileiro experiente. Considere linguagem sensacionalista, "
        "alegações verificáveis, ausência de fontes, apelos emocionais e pedidos de compartilhamento. "
        + _FORMAT,
        "Avalie a probabilidade (0-100) de este texto conter desinformação.\n\nTexto:\n{text}",
    ),
    "zero_cautious": (
        "Você avalia notícias em português brasileiro. Seja calibrado: use valores próximos de 50 "
        "quando houver incerteza genuína, e valores extremos apenas quando tiver forte convicção. "
        + _FORMAT,
        "Qual a probabilidade (0-100) de este texto conter desinformação?\n\nTexto:\n{text}",
    ),
}

_FORMAT_DIGIT = (
    "Responda APENAS com um único algarismo de 0 a 9, sem texto e sem explicação. "
    "0 = certamente informação legítima. 9 = certamente desinformação."
)
PROMPTS["digit_direct"] = (
    "Você avalia notícias em português brasileiro. " + _FORMAT_DIGIT,
    "Qual a probabilidade (0-9) de este texto conter desinformação?\n\nTexto:\n{text}",
)
PROMPTS["digit_analytic"] = (
    "Você é um verificador de fatos brasileiro experiente. Considere linguagem sensacionalista, "
    "alegações verificáveis, ausência de fontes e apelos emocionais. " + _FORMAT_DIGIT,
    "Avalie a probabilidade (0-9) de este texto conter desinformação.\n\nTexto:\n{text}",
)

FEW_SHOT_ARM = "few_shot"
DIGIT_FEW_SHOT_ARM = "digit_few_shot"
_SCORE_RE = re.compile(r"\d{1,3}")


def parse_score(text: str) -> float | None:
    """First integer in 0-100, as a probability."""
    for m in _SCORE_RE.finditer(text or ""):
        v = int(m.group())
        if 0 <= v <= 100:
            return v / 100.0
    return None


def build_few_shot(corpus: pd.DataFrame, sample: pd.DataFrame, seed: int, k: int = 4,
                   digit_scale: bool = False) -> str:
    """Few-shot examples drawn from rows NOT in the evaluation sample."""
    used = set(sample["text"].astype(str))
    pool = corpus[~corpus["text"].astype(str).isin(used)]
    if pool.empty:
        return ""
    parts = []
    for label, group in pool.groupby("label"):
        take = group.sample(min(k // 2, len(group)), random_state=seed)
        for _, row in take.iterrows():
            snippet = str(row["text"])[:600].replace("\n", " ")
            if digit_scale:
                answer = "8" if int(label) == 1 else "1"
            else:
                answer = "85" if int(label) == 1 else "15"
            parts.append(f"Texto: {snippet}\nResposta: {answer}")
    return "\n\n".join(parts)


def classify(records: list[str], client: LLMClient, system: str, user_tpl: str,
             workers: int, use_logprobs: bool = False) -> tuple[list[float | None], list[str]]:
    """Score every record, preserving input order."""
    scores: list[float | None] = [None] * len(records)
    errors: list[str] = []

    def one(i: int) -> tuple[int, float | None, str | None]:
        text = records[i][:MAX_CHARS]
        completion, err, top = client.complete(
            system, user_tpl.format(text=text), logprobs=use_logprobs, meta={"row": i}
        )
        if err:
            return i, None, err
        if use_logprobs:
            score = expected_digit(top)
            if score is None:
                raw = parse_score(completion)
                score = None if raw is None else min(1.0, raw * 100 / 9.0)
        else:
            score = parse_score(completion)
        return i, score, None if score is not None else f"unparseable: {completion!r}"

    pending = list(range(len(records)))
    errors: list[str] = []
    for attempt in range(3):
        if not pending:
            break
        if attempt:
            delay = retry_delay(attempt - 1)
            print(f"    retry {attempt}: {len(pending)} retryable row(s), waiting {delay:.1f}s")
            time.sleep(delay)
        failed, errors = [], []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for i, score, err in pool.map(one, pending):
                scores[i] = score
                if err:
                    errors.append(err)
                    if is_retryable(err):
                        failed.append(i)
        pending = failed
    return scores, errors


def report(name: str, y: np.ndarray, probs: np.ndarray) -> dict:
    pred = (probs >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", pos_label=1, zero_division=0
    )
    auc = float(roc_auc_score(y, probs)) if len(set(y.tolist())) > 1 else float("nan")
    thr, _, _, bf1 = best_f1_threshold(y, probs)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    row = {
        "arm": name, "n": len(y),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "roc_auc": auc, "pr_auc": float(average_precision_score(y, probs)),
        "best_thr": thr, "best_f1": float(bf1), "vs_trivial": float(bf1) - TRIVIAL_F1,
        "ece": float(ece(y, probs)),
        "distinct_scores": int(len(set(probs.tolist()))),
        "fp": int(cm[0, 1]), "fn": int(cm[1, 0]),
    }
    print(f"\n=== {name} (n={len(y)}) ===")
    print(f"  @0.5     accuracy {row['accuracy']:.4f} | precision {row['precision']:.4f} | "
          f"recall {row['recall']:.4f} | f1 {row['f1']:.4f}")
    print(f"  ranking  ROC-AUC {row['roc_auc']:.4f} | PR-AUC {row['pr_auc']:.4f}")
    print(f"  @best    threshold {thr:.4f} | f1 {row['best_f1']:.4f} "
          f"({row['vs_trivial']:+.4f} vs the trivial all-fake floor of {TRIVIAL_F1:.4f})")
    print(f"  ECE      {row['ece']:.4f}   <- is the number it reports believable?")
    print(f"  spread   {row['distinct_scores']} distinct scores over {len(y)} rows")
    print(f"  errors   FP {row['fp']} (legit flagged) / FN {row['fn']} (fake passed)")
    return row


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="LLM zero/few-shot classification baseline.")
    p.add_argument("--data", default="data/processed/fakerecogna2.csv")
    p.add_argument("--sample", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--arms", default="zero_direct")
    p.add_argument("--model", default=None, help="Override the config model (e.g. sabia-4).")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--repeats", type=int, default=1,
                   help="Run each arm this many times and report the spread. Hosted inference is "
                        "NOT deterministic even at temperature 0 (measured: 11/20 identical "
                        "completions across two runs), so a single draw understates uncertainty.")
    p.add_argument("--save-probs", action="store_true",
                   help="Write ml/results/eval/<corpus>__llm.csv, row-aligned with the other models. "
                        "Only meaningful at the full sample size the other models used.")
    p.add_argument("--tag", default="llm", help="Model name used in the saved probability filename.")
    p.add_argument("--probs-out", default=None,
                   help="Write row-aligned label,prob to this exact path instead of "
                        "<RESULTS_DIR>/eval/. Use for sub-full samples, which must NEVER land in "
                        "eval/ - the analyses there align files element-wise, so a 500-row file "
                        "beside the 5,000-row ones would silently corrupt every comparison.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    cfg = load_llm_config()
    model = args.model or cfg.get("model")
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[llm-eval] ERROR: {data_path} not found.")
        return 1

    corpus = pd.read_csv(data_path).dropna(subset=["text", "label"])
    corpus["label"] = corpus["label"].astype(int)
    sample = stratified_sample(corpus, args.sample, args.seed)
    texts = sample["text"].astype(str).tolist()
    y = sample["label"].to_numpy()
    print(f"[llm-eval] {data_path.name}: {len(sample)} rows "
          f"(fake={int(y.sum())}, true={int((y == 0).sum())}), model={model}")

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    shot_arms = {FEW_SHOT_ARM, DIGIT_FEW_SHOT_ARM}
    unknown = [a for a in arms if a not in PROMPTS and a not in shot_arms]
    if unknown:
        print(f"[llm-eval] ERROR: unknown arms {unknown}; "
              f"known: {list(PROMPTS) + sorted(shot_arms)}")
        return 1

    blocks = {}
    for arm in shot_arms & set(arms):
        digit = arm == DIGIT_FEW_SHOT_ARM
        blocks[arm] = build_few_shot(corpus, sample, args.seed, digit_scale=digit)
        if not blocks[arm]:
            print("[llm-eval] ERROR: could not build few-shot examples outside the evaluation "
                  "sample. Using in-sample examples would be leakage, so this fails rather than "
                  "falling back.")
            return 1

    def prompt_for(arm: str) -> tuple[str, str]:
        if arm not in shot_arms:
            return PROMPTS[arm]
        base = "digit_direct" if arm == DIGIT_FEW_SHOT_ARM else "zero_direct"
        system, user = PROMPTS[base]
        return system, f"Exemplos:\n\n{blocks[arm]}\n\n---\n\n{user}"

    est_in = est_out = 0
    for arm in arms:
        system, user_tpl = prompt_for(arm)
        overhead = approx_tokens(system) + approx_tokens(user_tpl.format(text=""))
        est_in += sum(approx_tokens(t[:MAX_CHARS]) + overhead for t in texts)
        est_out += len(texts) * int(cfg.get("max_tokens", 16))
    cost = estimate_cost(cfg, model, 1, est_in, est_out)
    total_calls = len(texts) * len(arms)
    print(f"[llm-eval] {total_calls} calls across {len(arms)} arm(s); "
          f"~{est_in:,} input + ~{est_out:,} output tokens; estimated R${cost:.2f}")

    if args.dry_run:
        print("\n[dry-run] no calls made. Example prompt for the first arm:\n")
        system, user_tpl = prompt_for(arms[0])
        print(f"  SYSTEM: {system}\n")
        print(f"  USER:   {user_tpl.format(text=texts[0][:300] + ' ...')[:900]}")
        return 0

    if not is_enabled():
        print("\n[llm-eval] MARITACA_API_KEY is not set, so no call was made. This is the offline "
              "default, not a result.")
        return 0

    rows, results_dir = [], Path(os.environ.get("RESULTS_DIR", "ml/results"))
    for arm, rep in [(a, r) for a in arms for r in range(args.repeats)]:
        label = arm if args.repeats == 1 else f"{arm}#{rep + 1}"
        system, user_tpl = prompt_for(arm)
        client = LLMClient(cfg=cfg, model=model,
                           run_name=f"{data_path.stem}__{arm}__{model}"
                                    + (f"__r{rep + 1}" if args.repeats > 1 else ""))
        scores, errors = classify(texts, client, system, user_tpl, args.workers,
                                  use_logprobs=arm.startswith("digit"))
        raw_path = client.save_raw()
        print(f"\n[{label}] {client.report()}" + (f" | raw -> {raw_path}" if raw_path else ""))
        if errors:
            print(f"[{label}] {len(errors)} failed; first: {errors[0][:160]}")

        ok = [i for i, s in enumerate(scores) if s is not None]
        if len(ok) < len(scores) * 0.9:
            print(f"[{label}] ABORTING this arm: only {len(ok)}/{len(scores)} usable responses. "
                  "Reporting metrics on a heavily filtered set would misrepresent the model.")
            client.close()
            continue
        probs = np.array([scores[i] for i in ok])
        row = report(f"{label} ({model})", y[ok], probs)
        row.update({"model": model, "corpus": data_path.stem, "repeat": rep + 1, "usable": len(ok),
                    "requested": len(scores), "cost_brl": client.usage.cost(cfg, model)})
        rows.append(row)

        if (args.save_probs or args.probs_out) and len(ok) == len(scores):
            if args.probs_out:
                out = Path(args.probs_out)
            else:
                out = results_dir / "eval" / f"{data_path.stem}__{args.tag}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"label": y, "prob": [scores[i] for i in range(len(scores))]}).to_csv(
                out, index=False)
            print(f"  saved    {out}  (row-aligned with the other models)")
        elif args.save_probs or args.probs_out:
            print("  NOT saving probabilities: some rows failed, so the file would not be "
                  "row-aligned with the other models and would silently corrupt every comparison.")
        client.close()

    if rows:
        results_dir.mkdir(parents=True, exist_ok=True)
        out = results_dir / "llm_baseline.csv"
        df = pd.DataFrame(rows)
        if out.exists():
            df = pd.concat([pd.read_csv(out), df], ignore_index=True)
        df.to_csv(out, index=False)
        print(f"\n[llm-eval] wrote {out}")
        print(f"[llm-eval] total spend this run: R${sum(r['cost_brl'] for r in rows):.4f}")
        if args.repeats > 1:
            print("\n  Run-to-run spread at temperature 0 (hosted inference is not deterministic):")
            fresh = pd.DataFrame(rows)
            fresh["base_arm"] = fresh["arm"].str.split("#").str[0]
            for base, g in fresh.groupby("base_arm"):
                if len(g) > 1:
                    print(f"    {base:<16} ROC-AUC {g['roc_auc'].mean():.4f} "
                          f"+/- {g['roc_auc'].std(ddof=1):.4f}   "
                          f"F1 {g['f1'].mean():.4f} +/- {g['f1'].std(ddof=1):.4f}   "
                          f"ECE {g['ece'].mean():.4f} +/- {g['ece'].std(ddof=1):.4f}")
            print("    Any comparison smaller than this spread is noise, not a result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
