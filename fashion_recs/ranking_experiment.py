"""Compare ranking objectives on identical chronological candidate groups."""
import gc
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from .expanded import ExpandedCandidateEngine
from .model import fit_pointwise
from .pipeline import bootstrap_difference, week_labels

OFFSETS = (63, 56, 49, 42)
HISTORY_DAYS = 98
LOOKBACK_DAYS = HISTORY_DAYS + max(OFFSETS) + 7
K = 12
WEIGHTS = np.array([0.35, 0.1, 3.0, 0.7, 1.0, 0.3])
METHODS = ("hybrid", "pointwise", "lambdamart")


@dataclass
class Queries:
    x: np.ndarray
    y: np.ndarray
    group: np.ndarray
    truth_sizes: np.ndarray
    cold: np.ndarray
    popularity_ap: np.ndarray


def make_queries(engine, labels):
    matrices, targets, group, truth_sizes, cold, popularity_ap = [], [], [], [], [], []
    popular = engine.popular[:K]
    for user, actual in labels.items():
        ids, features = engine.candidates(user)
        truth = set(actual)
        if not truth or not ids:
            raise ValueError("A query needs targets and candidates")
        y = np.fromiter((a in truth for a in ids), dtype=np.int8, count=len(ids))
        matches = np.fromiter((a in truth for a in popular), dtype=np.int8, count=len(popular))
        matrices.append(features)
        targets.append(y)
        group.append(len(ids))
        truth_sizes.append(len(truth))
        cold.append(user not in engine.histories)
        popularity_ap.append(ap_from_matches(matches, len(truth)))
    return Queries(np.concatenate(matrices), np.concatenate(targets), np.asarray(group, dtype=np.int32),
                   np.asarray(truth_sizes), np.asarray(cold), np.asarray(popularity_ap))


def ap_from_matches(matches, truth_size):
    matches = matches[:K]
    return float(np.sum(np.cumsum(matches) * matches / np.arange(1, len(matches) + 1)) / min(truth_size, K))


def fit_lambdamart(queries, seed):
    import lightgbm as lgb

    if queries.group.sum() != len(queries.y) or len(queries.x) != len(queries.y):
        raise ValueError("Ranking groups must cover the contiguous feature rows")
    dataset = lgb.Dataset(queries.x, label=queries.y, group=queries.group)
    return lgb.train({"objective": "lambdarank", "lambdarank_truncation_level": K + 3,
                      "learning_rate": 0.08, "num_leaves": 15, "max_depth": 5,
                      "min_data_in_leaf": 40, "lambda_l2": 3.0, "num_threads": 4,
                      "deterministic": True, "force_col_wise": True, "seed": seed,
                      "verbosity": -1}, dataset, num_boost_round=70)


def score_queries(queries, pointwise, lambdamart):
    values = (queries.x @ WEIGHTS, pointwise.predict_proba(queries.x)[:, 1],
              lambdamart.predict(queries.x, num_threads=4))
    if any(len(v) != len(queries.y) or not np.isfinite(v).all() for v in values):
        raise ValueError("Invalid ranking scores")
    ap = np.empty((len(queries.group), len(METHODS)))
    oracle, recall = [], []
    start = 0
    for i, size in enumerate(queries.group):
        stop = start + size
        relevance = queries.y[start:stop]
        truth_size = queries.truth_sizes[i]
        if queries.cold[i]:
            ap[i] = queries.popularity_ap[i]
            oracle.append(queries.popularity_ap[i])
        else:
            for j, scores in enumerate(values):
                order = np.argsort(-scores[start:stop], kind="stable")[:K]
                ap[i, j] = ap_from_matches(relevance[order], truth_size)
            oracle.append(min(int(relevance.sum()), K) / min(truth_size, K))
        recall.append(int(relevance.sum()) / truth_size)
        start = stop
    metrics = dict(zip(METHODS, ap.mean(axis=0).tolist()))
    return {"map_at_12": metrics, "popularity_map_at_12": float(queries.popularity_ap.mean()),
            "candidate_recall": float(np.mean(recall)), "oracle_map_at_12": float(np.mean(oracle)),
            "cold_users": int(queries.cold.sum()), "mean_candidates": float(queries.group.mean()),
            "paired_lambdamart_minus_hybrid": {"difference": float((ap[:, 2] - ap[:, 0]).mean()),
                                               "ci_95": bootstrap_difference(ap[:, 2], ap[:, 0])},
            "paired_lambdamart_minus_pointwise": {"difference": float((ap[:, 2] - ap[:, 1]).mean()),
                                                  "ci_95": bootstrap_difference(ap[:, 2], ap[:, 1])}}


def review_gate(weeks):
    paired = [w["paired_lambdamart_minus_hybrid"] for w in weeks]
    mean = float(np.mean([p["difference"] for p in paired]))
    wins = sum(p["difference"] > 0 for p in paired)
    return {"mean_delta": mean, "weeks_won": wins,
            "passes": bool(mean > 0 and wins >= 3 and paired[-1]["ci_95"][0] > 0)}


def run_ranking_comparison(transactions, articles, end, output, *, train_users=3000,
                           eval_users=5000, seed=42, source=None, synthetic=False):
    import lightgbm as lgb

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cutoffs = [pd.Timestamp(end) - pd.Timedelta(days=offset) for offset in OFFSETS]
    first_train = cutoffs[0] - pd.Timedelta(days=7)
    if transactions.t_dat.min() > first_train - pd.Timedelta(days=HISTORY_DAYS):
        raise ValueError(f"Need {LOOKBACK_DAYS} days of source history for the first feature window")
    previous = None
    weeks = []
    for cutoff in [first_train, *cutoffs]:
        print(f"{cutoff.date()}: shared candidate features", flush=True)
        history = transactions.loc[transactions.t_dat.ge(cutoff - pd.Timedelta(days=HISTORY_DAYS))
                                   & transactions.t_dat.lt(cutoff)]
        engine = ExpandedCandidateEngine().fit(history, articles, cutoff)
        feature_max = str(engine.max_seen_date.date())
        if previous is not None:
            training, training_start, train_feature_max = previous
            if training_start + pd.Timedelta(days=7) != cutoff:
                raise ValueError("Training labels must end at the evaluation cutoff")
            pointwise = fit_pointwise(training.x, training.y, seed)
            lambdamart = fit_lambdamart(training, seed)
            labels, population = week_labels(transactions, cutoff, eval_users, seed)
            evaluation = make_queries(engine, labels)
            metrics = score_queries(evaluation, pointwise, lambdamart)
            week = {"start": str(cutoff.date()), "end_exclusive": str((cutoff + pd.Timedelta(days=7)).date()),
                    "training_labels_start": str(training_start.date()), "training_feature_max_date": train_feature_max,
                    "feature_max_date": feature_max, "training_users": len(training.group),
                    "training_pairs": len(training.y), "training_positive_pairs": int(training.y.sum()),
                    "evaluated_users": len(evaluation.group), "population": population, **metrics}
            weeks.append(week)
            print(json.dumps(week), flush=True)
            (output / "ranking.partial.json").write_text(json.dumps(weeks, indent=2), encoding="utf-8")
            del evaluation, labels, pointwise, lambdamart, training, previous
        if cutoff != cutoffs[-1]:
            train_labels, _ = week_labels(transactions, cutoff, train_users, seed)
            previous = (make_queries(engine, train_labels), cutoff, feature_max)
            del train_labels
        del engine, history
        gc.collect()
    report = {"data_kind": "SYNTHETIC DEMO" if synthetic else "H&M competition data",
              "seed": seed, "history_days": HISTORY_DAYS, "source": source or {}, "weeks": weeks,
              "rule": "Positive average delta, at least three weekly wins, and positive lower paired interval on the last week.",
              "review_gate": review_gate(weeks),
              "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                          "pandas": pd.__version__, "sklearn": sklearn.__version__, "lightgbm": lgb.__version__},
              "elapsed_seconds": round(time.perf_counter() - started, 2)}
    (output / "ranking.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["review_gate"]), flush=True)
    return report
