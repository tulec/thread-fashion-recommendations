import argparse
import hashlib
import json
import pickle
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from .data import load_data, synthetic_data
from .metrics import apk
from .model import CandidateEngine, fit_ranker


def week_labels(transactions, start, max_users, seed):
    frame = transactions.loc[transactions.t_dat.ge(start) & transactions.t_dat.lt(start + pd.Timedelta(days=7))]
    labels = frame.groupby("customer_id").article_id.agg(lambda s: list(dict.fromkeys(s))).to_dict()
    users = sorted(labels)
    if max_users and len(users) > max_users:
        selected = np.random.default_rng(seed).choice(users, size=max_users, replace=False)
        labels = {user: labels[user] for user in sorted(selected)}
    return labels, int(frame.customer_id.nunique())


def bootstrap_difference(candidate, baseline, seed=42):
    diff = np.asarray(candidate) - np.asarray(baseline)
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(diff, size=len(diff), replace=True).mean()) for _ in range(500)]
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def evaluate(engine, ranker, labels):
    per_user = {name: [] for name in ["popular", "hybrid", "learned"]}
    recalls, catalogue = [], set()
    cold_count, cold_scores, warm_scores = 0, [], []
    for user, actual in labels.items():
        candidates, _ = engine.candidates(user)
        recalls.append(len(set(actual) & set(candidates)) / len(set(actual)))
        for name in per_user:
            recommendations = engine.recommend(user, name, ranker)
            score = apk(actual, recommendations)
            per_user[name].append(score)
            if name == "learned":
                catalogue.update(recommendations)
                if user not in engine.histories:
                    cold_count += 1
                    cold_scores.append(score)
                else:
                    warm_scores.append(score)
    metrics = {name: float(np.mean(scores)) for name, scores in per_user.items()}
    return {
        "map_at_12": metrics,
        "learned_minus_popular_95pct_bootstrap_ci": bootstrap_difference(per_user["learned"], per_user["popular"]),
        "candidate_recall": float(np.mean(recalls)),
        "catalogue_coverage": len(catalogue) / len(engine.popular),
        "evaluated_users": len(labels), "cold_users": cold_count,
        "cold_map_at_12": float(np.mean(cold_scores)) if cold_scores else None,
        "warm_map_at_12": float(np.mean(warm_scores)) if warm_scores else None,
    }


def run(transactions, articles, end, output, *, synthetic, train_users=3000, eval_users=5000, seed=42, source=None,
        engine_class=CandidateEngine, serving_method="learned", history_days=None):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    training_start = end - pd.Timedelta(days=14)
    evaluation_start = end - pd.Timedelta(days=7)
    training_labels, total_train = week_labels(transactions, training_start, train_users, seed)
    evaluation_labels, total_eval = week_labels(transactions, evaluation_start, eval_users, seed)
    if not training_labels or not evaluation_labels:
        raise ValueError("At least two complete, non-empty label weeks are required")
    print(f"Training labels: {len(training_labels):,}; evaluation: {len(evaluation_labels):,}", flush=True)
    if serving_method not in {"learned", "hybrid"}:
        raise ValueError("Serving method must be learned or hybrid")
    if history_days is not None and history_days <= 0:
        raise ValueError("History days must be positive")
    def window(cutoff):
        if history_days is None:
            return transactions
        return transactions.loc[transactions.t_dat.ge(cutoff - pd.Timedelta(days=history_days))]
    training_engine = engine_class().fit(window(training_start), articles, training_start)
    training_feature_max_date = str(training_engine.max_seen_date.date())
    ranker, training_info = fit_ranker(training_engine, training_labels, seed)
    del training_engine
    evaluation_engine = engine_class().fit(window(evaluation_start), articles, evaluation_start)
    metrics = evaluate(evaluation_engine, ranker, evaluation_labels)
    report = {
        "data_kind": "SYNTHETIC DEMO — NOT H&M RESULTS" if synthetic else "H&M competition data",
        "seed": seed, "k": 12, "training_labels_start": str(training_start.date()),
        "evaluation_labels_start": str(evaluation_start.date()), "evaluation_labels_end_exclusive": str(end.date()),
        "evaluation_feature_max_date": str(evaluation_engine.max_seen_date.date()),
        "training_feature_max_date": training_feature_max_date,
        "engine_class": engine_class.__name__, "serving_method": serving_method,
        "history_days": history_days,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "training_population": total_train, "evaluation_population": total_eval,
        "sampling": "Seeded uniform sample of customers purchasing in each label week; includes cold customers. Zero means all.",
        "ranking_training": training_info, "source": source or {}, **metrics,
        "hyperparameter_policy": "Fixed parameters before holdout evaluation; no holdout tuning or model selection.",
    }
    del evaluation_engine
    # Fit serving features on all observed history, keeping the learned ranker fixed.
    final_engine = engine_class().fit(window(end), articles, end)
    article_rows = articles.fillna("").to_dict(orient="records")
    metadata = {row["article_id"]: row for row in article_rows}
    demo_users = list(evaluation_labels)[:100]
    payload = {
        "data_kind": report["data_kind"], "as_of": str(end.date()), "metrics": metrics,
        "serving_method": serving_method, "engine_class": engine_class.__name__,
        "users": [], "popular": final_engine.popular[:12],
    }
    visible_ids = set(final_engine.popular[:12])
    for index, user in enumerate(demo_users, 1):
        recs = final_engine.recommend(user, serving_method, ranker)
        history = [a for a, _, _ in final_engine.histories.get(user, [])]
        candidates, features = final_engine.candidates(user)
        feature_lookup = dict(zip(candidates, features))
        reasons = {}
        for article in recs:
            f = feature_lookup[article]
            reasons[article] = (
                "Покупали раньше" if f[2] > 0 else
                "Похожая категория покупок" if f[4] > 0 else
                "Связь с прошлыми покупками" if f[5] > 0 else "Популярно сейчас"
            )
        # Export opaque demo labels rather than actual Kaggle customer identifiers.
        payload["users"].append({"id": f"Покупатель {index:03d}", "history": history, "recommendations": recs, "reasons": reasons})
        visible_ids.update(recs + history)
    payload["articles"] = {a: metadata.get(a, {"article_id": a}) for a in sorted(visible_ids)}
    report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output / "demo.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")
    with (output / "model.pkl").open("wb") as stream:
        pickle.dump({"engine": final_engine, "ranker": ranker, "synthetic": synthetic,
                     "serving_method": serving_method}, stream)
    kind = report["data_kind"]
    scores = metrics["map_at_12"]
    text = f"""# Evaluation report\n\nData: **{kind}**\n\nTraining labels: {training_start.date()} to {evaluation_start.date()} (exclusive).\nHeld-out labels: {evaluation_start.date()} to {end.date()} (exclusive).\nAll evaluation features predate {evaluation_start.date()}.\n\n| Model | MAP@12 |\n|---|---:|\n| Recent popularity | {scores['popular']:.6f} |\n| Fixed hybrid | {scores['hybrid']:.6f} |\n| Learned candidate scorer | {scores['learned']:.6f} |\n\nEvaluated customers: {len(evaluation_labels)} / {total_eval}.\nCandidate recall: {metrics['candidate_recall']:.4f}.\nCold customers: {metrics['cold_users']}.\n\nSee report.json for the paired bootstrap interval and complete configuration.\nThese are offline results among purchasers in one held-out week, not online business uplift.\nCandidates come only from prior purchases: genuinely new catalogue items need a separate content candidate generator.\nReason badges describe available signals, not causal model explanations.\nNo official Kaggle score is claimed.\n"""
    (output / "report.md").write_text(text, encoding="utf-8")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file() and p.name != "manifest.json"}
    (output / "manifest.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print(json.dumps({"data_kind": kind, **metrics, "output": str(output)}, ensure_ascii=False, indent=2), flush=True)
    return report


def submit(model_path, sample_path, output):
    # Only load locally produced pickle files. Never load an untrusted pickle.
    with model_path.open("rb") as stream:
        model = pickle.load(stream)
    if model["synthetic"]:
        raise ValueError("Synthetic models cannot generate a competition submission")
    engine, ranker = model["engine"], model["ranker"]
    if len(engine.popular) < 12:
        raise ValueError("At least 12 observed articles are required")
    output.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["customer_id", "prediction"])
        for chunk in pd.read_csv(sample_path, usecols=["customer_id"], dtype="string", chunksize=10000):
            for user in chunk.customer_id:
                recommendations = engine.recommend(user, model.get("serving_method", "learned"), ranker)
                if len(recommendations) != 12 or len(set(recommendations)) != 12:
                    raise ValueError("Submission requires 12 unique articles per customer")
                writer.writerow([user, " ".join(recommendations)])
    print(f"Created {output}; not uploaded to Kaggle", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a fashion recommender")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["demo", "train"]:
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, default=Path("artifacts") / name)
        command.add_argument("--train-users", type=int, default=3000)
        command.add_argument("--eval-users", type=int, default=5000)
        command.add_argument("--seed", type=int, default=42)
        if name == "train":
            command.add_argument("--data-dir", type=Path, required=True)
    command = commands.add_parser("submit")
    command.add_argument("--model", type=Path, required=True)
    command.add_argument("--sample", type=Path, required=True)
    command.add_argument("--output", type=Path, default=Path("artifacts/submission.csv"))
    args = parser.parse_args()
    if args.command == "submit":
        submit(args.model, args.sample, args.output)
        return
    if args.train_users < 0 or args.eval_users < 0:
        parser.error("User limits must be non-negative; zero means all")
    if args.command == "demo":
        tx, articles, end = synthetic_data(args.seed)
        source = {"generator": "fashion_recs.data.synthetic_data", "rows": len(tx)}
    else:
        tx, articles, end, source = load_data(args.data_dir)
    run(tx, articles, end, args.output, synthetic=args.command == "demo", train_users=args.train_users,
        eval_users=args.eval_users, seed=args.seed, source=source)


if __name__ == "__main__":
    main()
