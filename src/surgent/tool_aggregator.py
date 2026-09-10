"""Probabilistic tool-label aggregator using product-of-experts.

Learns reverse conditionals  θ_m(a|b) = P(T=a | ŷ_m=b)  from labeled dev
data, then infers the most-likely true label on unlabeled test data by
combining all model predictions with a Bayesian product-of-experts rule.

Usage (CLI)
-----------
    python -m surgent.tool_aggregator \
        --dev   dev.jsonl   \
        --test  test.jsonl  \
        --out   predictions.jsonl \
        --alpha 1.0 --alpha0 1.0 \
        --threshold 0.4 \
        --fallback prior \
        --weights "gemini-2.5-flash=1.2,claude-opus-4-6=0.8"

JSONL schemas
-------------
Dev  line:  {"id": str, "gt": str, "predictions": {model: label, ...}}
Test line:  {"id": str, "predictions": {model: label, ...}}
Out  line:  {"id": str, "pred": str, "posterior": {label: float},
             "top2": [str, str], "max_prob": float}

The module is also importable — see ``ToolAggregator`` class.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Core aggregator
# ---------------------------------------------------------------------------

class ToolAggregator:
    """Product-of-experts label aggregator with Dirichlet smoothing."""

    def __init__(
        self,
        alpha: float = 1.0,
        alpha0: float = 1.0,
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.0,
        fallback: str = "prior",  # "prior" or "uniform"
    ):
        self.alpha = alpha
        self.alpha0 = alpha0
        self.weights = weights or {}
        self.threshold = threshold
        self.fallback = fallback

        # Learned parameters (populated by .fit())
        self.labels: List[str] = []
        self.models: List[str] = []
        self.prior: Dict[str, float] = {}           # π(a)   log-space
        self.theta: Dict[str, Dict[str, Dict[str, float]]] = {}  # θ[m][b][a] log-space

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        dev_items: Sequence[Dict[str, Any]],
        label_set: Optional[List[str]] = None,
    ) -> "ToolAggregator":
        """Learn θ_m(a|b) and π(a) from labeled dev items.

        Parameters
        ----------
        dev_items : list of dicts
            Each dict has ``"gt"`` (str) and ``"predictions"`` ({model: label}).
        label_set : list of str, optional
            Explicit label vocabulary.  If None, inferred from data.
        """
        # --- discover labels and models ---
        gt_counter: Counter = Counter()
        model_set: set = set()
        for item in dev_items:
            gt_counter[item["gt"]] += 1
            model_set.update(item["predictions"].keys())

        if label_set is not None:
            self.labels = list(label_set)
        else:
            self.labels = sorted(gt_counter.keys())
        self.models = sorted(model_set)

        L = len(self.labels)
        label_idx = {a: i for i, a in enumerate(self.labels)}

        # --- prior π(a) ---
        N_total = sum(gt_counter.values())
        self.prior = {}
        for a in self.labels:
            self.prior[a] = math.log(
                (gt_counter.get(a, 0) + self.alpha0) / (N_total + self.alpha0 * L)
            )

        # --- reverse conditionals θ_m(a|b) ---
        # N_m[b][a] = count of (gt=a, pred_m=b)
        N_m: Dict[str, Dict[str, Counter]] = {
            m: defaultdict(Counter) for m in self.models
        }
        for item in dev_items:
            gt = item["gt"]
            if gt not in label_idx:
                continue
            for m, b in item["predictions"].items():
                if m in model_set:
                    N_m[m][b][gt] += 1

        self.theta = {}
        for m in self.models:
            self.theta[m] = {}
            for b in N_m[m]:
                N_dot_b = sum(N_m[m][b].values())
                self.theta[m][b] = {}
                for a in self.labels:
                    self.theta[m][b][a] = math.log(
                        (N_m[m][b].get(a, 0) + self.alpha)
                        / (N_dot_b + self.alpha * L)
                    )

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_one(
        self,
        predictions: Dict[str, str],
    ) -> Dict[str, Any]:
        """Predict the true label for a single test item.

        Parameters
        ----------
        predictions : {model_id: predicted_label}

        Returns
        -------
        dict with keys: pred, posterior, top2, max_prob
        """
        L = len(self.labels)
        log_uniform = -math.log(L) if L > 0 else 0.0

        logscores: Dict[str, float] = {}
        for a in self.labels:
            ls = self.prior[a]
            for m, b in predictions.items():
                if m not in self.theta:
                    continue  # unknown model — skip
                w = self.weights.get(m, 1.0)
                if b in self.theta[m]:
                    ls += w * self.theta[m][b][a]
                else:
                    # unseen predicted label: back off
                    if self.fallback == "prior":
                        ls += w * self.prior[a]
                    else:
                        ls += w * log_uniform
            logscores[a] = ls

        # --- normalize (log-sum-exp) ---
        posterior = _softmax(logscores)

        # --- decision ---
        ranked = sorted(posterior.items(), key=lambda kv: -kv[1])
        pred = ranked[0][0]
        max_prob = ranked[0][1]
        top2 = [ranked[i][0] for i in range(min(2, len(ranked)))]

        if max_prob < self.threshold:
            pred = "UNCERTAIN"

        return {
            "pred": pred,
            "posterior": {k: round(v, 6) for k, v in posterior.items()},
            "top2": top2,
            "max_prob": round(max_prob, 6),
        }

    def predict(
        self,
        test_items: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Predict on a batch of test items.

        Each item must have ``"id"`` and ``"predictions"``.
        """
        results = []
        for item in test_items:
            out = self.predict_one(item["predictions"])
            out["id"] = item["id"]
            results.append(out)
        return results

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable summary of learned parameters."""
        lines = [
            f"ToolAggregator  α={self.alpha}  α0={self.alpha0}  "
            f"fallback={self.fallback}  τ={self.threshold}",
            f"  Labels ({len(self.labels)}): {self.labels}",
            f"  Models ({len(self.models)}): {self.models}",
            "",
            "  Prior π(a):",
        ]
        for a in self.labels:
            lines.append(f"    {a:<20s}  {math.exp(self.prior[a]):.4f}")

        for m in self.models:
            w = self.weights.get(m, 1.0)
            lines.append(f"\n  θ_{m}  (weight={w:.2f}):")
            # collect all observed predicted labels for this model
            pred_labels = sorted(self.theta[m].keys())
            col_header = "pred \\ gt"
            header = f"    {col_header:<20s}" + "".join(
                f"  {a:<12s}" for a in self.labels
            )
            lines.append(header)
            for b in pred_labels:
                row = f"    {b:<20s}"
                for a in self.labels:
                    row += f"  {math.exp(self.theta[m][b][a]):12.4f}"
                lines.append(row)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(logscores: Dict[str, float]) -> Dict[str, float]:
    """Numerically stable softmax over a dict of log-scores."""
    if not logscores:
        return {}
    max_ls = max(logscores.values())
    exps = {k: math.exp(v - max_ls) for k, v in logscores.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def items_from_results(
    all_results: Dict[str, list],
    *,
    absent_label: str = "(absent)",
) -> List[Dict[str, Any]]:
    """Convert the notebook's ``all_results`` dict to aggregator items.

    ``all_results`` is  {model_id: [result_dict, ...]}.  Each result_dict
    has keys: video_id, criterion, granularity, start_frame, end_frame,
    gt_present, gt_tool_types, pred_present, pred_tool_type.

    Returns a list of items with ``id``, ``gt``, ``predictions``.
    Multiple GT tool types are collapsed to the first entry.
    """
    # Build an index of clips  (keyed by a stable id)
    clips: Dict[str, Dict[str, Any]] = {}
    model_ids = list(all_results.keys())

    for model_id in model_ids:
        for r in all_results[model_id]:
            clip_id = (
                f"{r['video_id']}__{r['criterion']}__{r['granularity']}"
                f"__{r['start_frame']}_{r['end_frame']}"
            )
            if clip_id not in clips:
                # GT: collapse multi-label to first; absent if not present
                if r["gt_present"] and r["gt_tool_types"]:
                    gt = r["gt_tool_types"][0]
                else:
                    gt = absent_label
                clips[clip_id] = {"id": clip_id, "gt": gt, "predictions": {}}

            # Prediction: map pred_present=False -> absent
            if r["pred_present"]:
                pred_label = r["pred_tool_type"]
            else:
                pred_label = absent_label

            clips[clip_id]["predictions"][model_id] = pred_label

    return list(clips.values())


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(items: Sequence[Dict[str, Any]], path: str | Path) -> None:
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_weights(s: str) -> Dict[str, float]:
    """Parse 'model1=1.2,model2=0.8' into a dict."""
    if not s:
        return {}
    weights = {}
    for pair in s.split(","):
        pair = pair.strip()
        if "=" in pair:
            k, v = pair.split("=", 1)
            weights[k.strip()] = float(v.strip())
    return weights


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Probabilistic tool-label aggregator (product-of-experts)."
    )
    parser.add_argument("--dev", required=True, help="Dev JSONL (with gt)")
    parser.add_argument("--test", required=True, help="Test JSONL (predictions only)")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="Dirichlet smoothing for θ_m(a|b) (default: 1.0)",
    )
    parser.add_argument(
        "--alpha0", type=float, default=1.0,
        help="Dirichlet smoothing for prior π(a) (default: 1.0)",
    )
    parser.add_argument(
        "--weights", type=str, default="",
        help="Model weights, e.g. 'model1=1.2,model2=0.8'",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.0,
        help="Uncertainty threshold τ (default: 0.0 = disabled)",
    )
    parser.add_argument(
        "--fallback", choices=["prior", "uniform"], default="prior",
        help="Backoff for unseen predicted labels (default: prior)",
    )
    parser.add_argument(
        "--labels", type=str, default="",
        help="Comma-separated label set (optional; inferred from dev if empty)",
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args(argv)

    dev_items = read_jsonl(args.dev)
    test_items = read_jsonl(args.test)
    weights = parse_weights(args.weights)
    label_set = [s.strip() for s in args.labels.split(",") if s.strip()] or None

    agg = ToolAggregator(
        alpha=args.alpha,
        alpha0=args.alpha0,
        weights=weights,
        threshold=args.threshold,
        fallback=args.fallback,
    )
    agg.fit(dev_items, label_set=label_set)

    if args.verbose:
        print(agg.summary(), file=sys.stderr)

    results = agg.predict(test_items)
    write_jsonl(results, args.out)

    print(
        f"Wrote {len(results)} predictions to {args.out}  "
        f"(dev={len(dev_items)}, labels={len(agg.labels)}, models={len(agg.models)})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
