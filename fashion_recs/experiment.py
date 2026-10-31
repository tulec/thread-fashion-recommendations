"""Paired rolling-week comparison with a frozen final confirmation week."""
import gc
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from .expanded import ExpandedCandidateEngine
from .metrics import apk
from .model import CandidateEngine, fit_ranker
from .pipeline import bootstrap_difference, week_labels


def score_variant(engine, ranker, labels):
    """Score each customer's feature matrix once for all three methods."""
    scores = {name: [] for name in ["popular", "hybrid", "learned"]}
    recall, sizes = [], []
    cold = []
    weights = np.array([0.35, 0.1, 3.0, 0.7, 1.0, 0.3])
    for user, actual in labels.items():
        ids, features = engine.candidates(user)
        truth = set(actual)
        recall.append(len(truth.intersection(ids)) / len(truth))
        sizes.append(len(ids))
        cold.append(user not in engine.histories)
        popular = engine.popular[:12]
        if cold[-1]:
            recommendations = {method: popular for method in scores}
        else:
            hybrid = features @ weights
            learned = ranker.predict_proba(features)[:, 1]
            recommendations = {"popular": popular,
                "hybrid": [ids[i] for i in np.argsort(-hybrid, kind="stable")[:12]],
                "learned": [ids[i] for i in np.argsort(-learned, kind="stable")[:12]]}
        for method, recs in recommendations.items():
            scores[method].append(apk(actual, recs))
    metrics = {method: float(np.mean(values)) for method, values in scores.items()}
    mask = np.asarray(cold)
    return {"map_at_12": metrics, "candidate_recall": float(np.mean(recall)),
            "mean_candidates": float(np.mean(sizes)), "cold_users": int(mask.sum()),
            "cold_map_at_12": float(np.mean(np.asarray(scores["learned"])[mask])) if mask.any() else None,
            "warm_map_at_12": float(np.mean(np.asarray(scores["learned"])[~mask])) if (~mask).any() else None}, scores


def run_comparison(transactions, articles, end, output, *, train_users=3000,
                   eval_users=5000, seed=42, synthetic=False, source=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = []
    # Three earlier development windows; the last is a confirmation of a period
    # already examined in v1. No claim of a newly unseen final test is made.
    for offset in [28, 21, 14, 7]:
        cutoff = pd.Timestamp(end) - pd.Timedelta(days=offset)
        training_start = cutoff - pd.Timedelta(days=7)
        train_labels, train_population = week_labels(transactions, training_start, train_users, seed)
        labels, population = week_labels(transactions, cutoff, eval_users, seed)
        if not train_labels or not labels:
            raise ValueError("Four non-empty evaluation weeks and their preceding training weeks are required")
        role = "previously_seen_confirmation" if offset == 7 else "development"
        week = {"start": str(cutoff.date()), "end_exclusive": str((cutoff + pd.Timedelta(days=7)).date()),
                "role": role, "evaluated_users": len(labels), "population": population,
                "training_users": len(train_labels), "training_population": train_population,
                "training_labels_start": str(training_start.date()), "variants": {}}
        predictions = {}
        for name, cls in [("original", CandidateEngine), ("expanded", ExpandedCandidateEngine)]:
            print(f"{cutoff.date()} {name}: training features", flush=True)
            train_history = transactions.loc[transactions.t_dat.ge(training_start - pd.Timedelta(days=98))
                                             & transactions.t_dat.lt(training_start)]
            training_engine = cls().fit(train_history, articles, training_start)
            train_max = str(training_engine.max_seen_date.date())
            ranker, train_info = fit_ranker(training_engine, train_labels, seed)
            del training_engine, train_history
            gc.collect()
            print(f"{cutoff.date()} {name}: evaluation features", flush=True)
            evaluation_history = transactions.loc[transactions.t_dat.ge(cutoff - pd.Timedelta(days=98))
                                                  & transactions.t_dat.lt(cutoff)]
            engine = cls().fit(evaluation_history, articles, cutoff)
            metrics, scores = score_variant(engine, ranker, labels)
            metrics.update({"feature_max_date": str(engine.max_seen_date.date()),
                            "training_feature_max_date": train_max, "training": train_info})
            week["variants"][name] = metrics
            predictions[name] = scores
            print(json.dumps({"week": week["start"], "variant": name, **metrics}, ensure_ascii=False), flush=True)
            del engine, ranker, evaluation_history
            gc.collect()
        week["paired_expanded_minus_original"] = {
            method: {"difference": float(np.mean(predictions["expanded"][method]) - np.mean(predictions["original"][method])),
                     "ci_95": bootstrap_difference(predictions["expanded"][method], predictions["original"][method], seed)}
            for method in ["hybrid", "learned"]
        }
        results.append(week)
        # Preserve completed windows even if a later computation is interrupted.
        (output / "comparison.partial.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    development = results[:-1]
    # A concrete selection rule, fixed in code before looking at these results.
    candidates = [(variant, method) for variant in ["original", "expanded"] for method in ["hybrid", "learned"]]
    selected = max(candidates, key=lambda pair: np.mean([w["variants"][pair[0]]["map_at_12"][pair[1]] for w in development]))
    report = {"data_kind": "SYNTHETIC DEMO" if synthetic else "H&M competition data",
              "seed": seed, "history_days": 98, "train_users_limit": train_users, "eval_users_limit": eval_users,
              "source": source or {}, "weeks": results,
              "selection_policy": "Highest unweighted mean MAP@12 across three development weeks; final week excluded.",
              "selected_variant": selected[0], "selected_method": selected[1],
              "confirmation_map_at_12": results[-1]["variants"][selected[0]]["map_at_12"][selected[1]],
              "confirmation_caveat": "2020-09-16 to 2020-09-22 was evaluated in v1; it is not a newly unseen test.",
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
              "elapsed_seconds": round(time.perf_counter() - started, 2)}
    (output / "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"selected": selected, "confirmation_map_at_12": report["confirmation_map_at_12"]}), flush=True)
    return report
