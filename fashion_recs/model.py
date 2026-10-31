from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

FEATURES = ["popularity_7d", "popularity_28d", "repeat_recency", "repeat_count", "category_affinity", "transition_score"]


class CandidateEngine:
    """Recent popularity + repeat purchases + consecutive-purchase neighbours.

    Every statistic uses dates strictly before as_of. The neighbour graph is
    a lightweight item-to-item signal, not a causal claim about a shopping basket.
    """

    def __init__(self, popular_candidates=60, neighbours=12, history_items=12):
        self.popular_candidates = popular_candidates
        self.neighbours = neighbours
        self.history_items = history_items

    def fit(self, transactions, articles, as_of):
        self.as_of = pd.Timestamp(as_of)
        frame = transactions.loc[transactions.t_dat.lt(self.as_of)].copy()
        if frame.empty:
            raise ValueError("No history strictly before cutoff")
        self.max_seen_date = frame.t_dat.max()
        self.categories = articles.set_index("article_id").product_type_name.fillna("Unknown").to_dict()
        self.count7 = frame.loc[frame.t_dat.ge(self.as_of - pd.Timedelta(days=7)), "article_id"].value_counts().to_dict()
        self.count28 = frame.loc[frame.t_dat.ge(self.as_of - pd.Timedelta(days=28)), "article_id"].value_counts().to_dict()
        counts = frame.article_id.value_counts().to_dict()
        self.popular = sorted(counts, key=lambda a: (-self.count7.get(a, 0), -self.count28.get(a, 0), -counts[a], a))
        personal = frame.groupby(["customer_id", "article_id"], observed=True).agg(
            last=("t_dat", "max"), count=("article_id", "size")
        ).reset_index().sort_values(["customer_id", "last", "article_id"], ascending=[True, False, True])
        personal = personal.groupby("customer_id", sort=False).head(self.history_items)
        self.histories = defaultdict(list)
        self.affinities = defaultdict(Counter)
        for row in personal.itertuples(index=False):
            age = (self.as_of - row.last).days
            self.histories[row.customer_id].append((row.article_id, age, int(row.count)))
            self.affinities[row.customer_id][self.categories.get(row.article_id, "Unknown")] += int(row.count)
        # Bound graph construction: at most six recent events per customer.
        graph_frame = frame.loc[frame.t_dat.ge(self.as_of - pd.Timedelta(days=28))].sort_values(
            ["customer_id", "t_dat", "article_id"], kind="stable"
        ).groupby("customer_id", sort=False).tail(6).copy()
        graph_frame["previous"] = graph_frame.groupby("customer_id").article_id.shift()
        pairs = graph_frame.loc[graph_frame.previous.notna() & graph_frame.previous.ne(graph_frame.article_id)]
        forward = pairs[["previous", "article_id"]].rename(columns={"previous": "source", "article_id": "target"})
        reverse = forward.rename(columns={"source": "target", "target": "source"})
        edges = pd.concat([forward, reverse]).groupby(["source", "target"]).size().rename("count").reset_index()
        edges = edges.sort_values(["source", "count", "target"], ascending=[True, False, True])
        edges = edges.groupby("source", sort=False).head(self.neighbours)
        self.graph = defaultdict(list)
        for row in edges.itertuples(index=False):
            self.graph[row.source].append((row.target, int(row.count)))
        return self

    def candidates(self, user):
        history = self.histories.get(user, [])
        personal = {article: (age, count) for article, age, count in history}
        neighbours = Counter()
        for article, age, _ in history[:6]:
            for neighbour, count in self.graph.get(article, []):
                neighbours[neighbour] += np.exp(-age / 28) * count
        ids = list(dict.fromkeys(self.popular[:self.popular_candidates] + list(personal) + list(neighbours)))
        affinity = self.affinities.get(user, {})
        denominator = max(sum(affinity.values()), 1)
        matrix = np.array([
            [np.log1p(self.count7.get(a, 0)), np.log1p(self.count28.get(a, 0)),
             np.exp(-personal[a][0] / 28) if a in personal else 0,
             np.log1p(personal[a][1]) if a in personal else 0,
             affinity.get(self.categories.get(a, "Unknown"), 0) / denominator,
             np.log1p(neighbours.get(a, 0))]
            for a in ids
        ], dtype=np.float32)
        return ids, matrix

    def recommend(self, user, method="hybrid", ranker=None, k=12):
        if method == "popular":
            return self.popular[:k]
        if method not in {"hybrid", "learned"}:
            raise ValueError(f"Unknown method: {method}")
        ids, features = self.candidates(user)
        # A cold customer has no personal signal; popularity is an explicit fallback.
        if not self.histories.get(user):
            return self.popular[:k]
        if method == "learned":
            if ranker is None:
                raise ValueError("A fitted ranker is required")
            scores = ranker.predict_proba(features)[:, 1]
        else:
            scores = features @ np.array([0.35, 0.1, 3.0, 0.7, 1.0, 0.3])
        order = np.argsort(-scores, kind="stable")[:k]
        return [ids[i] for i in order]


def fit_ranker(engine, labels, seed=42):
    rows, targets = [], []
    for user, actual in labels.items():
        ids, matrix = engine.candidates(user)
        rows.append(matrix)
        truth = set(actual)
        targets.extend(int(item in truth) for item in ids)
    x = np.concatenate(rows)
    y = np.asarray(targets)
    if len(np.unique(y)) < 2:
        raise ValueError("Training candidates contain no positive examples; increase training users")
    # Fixed iteration count; sklearn's default random validation split is disabled.
    ranker = HistGradientBoostingClassifier(
        max_iter=70, max_leaf_nodes=15, max_depth=5, min_samples_leaf=40,
        learning_rate=0.08, l2_regularization=3.0, early_stopping=False, random_state=seed,
    )
    ranker.fit(x, y, sample_weight=np.where(y == 1, 15.0, 1.0))
    return ranker, {"rows": len(y), "positives": int(y.sum()), "features": FEATURES}

