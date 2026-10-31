"""Additional candidate sources, using only purchases before the cutoff."""
from collections import defaultdict

import numpy as np
import pandas as pd

from .model import CandidateEngine


class ExpandedCandidateEngine(CandidateEngine):
    def __init__(self, popular_candidates=120, neighbours=12, history_items=24,
                 category_items=40, category_count=2, co_purchase_neighbours=24,
                 max_candidates=480):
        super().__init__(popular_candidates, neighbours, history_items)
        self.category_items = category_items
        self.category_count = category_count
        self.co_purchase_neighbours = co_purchase_neighbours
        self.max_candidates = max_candidates

    def fit(self, transactions, articles, as_of):
        super().fit(transactions, articles, as_of)
        self.category_popular = defaultdict(list)
        self.variants = defaultdict(list)
        catalogue = articles.set_index("article_id")
        self.product_codes = (
            catalogue.product_code.astype(str).to_dict() if "product_code" in catalogue
            else {a: a[:-3] for a in catalogue.index}
        )
        # Only articles actually observed before as_of are eligible. Catalogue
        # metadata alone cannot make a future, unsold item a candidate.
        for article in self.popular:
            category = self.categories.get(article, "Unknown")
            if len(self.category_popular[category]) < self.category_items:
                self.category_popular[category].append(article)
            code = self.product_codes.get(article, article)
            if len(self.variants[code]) < 6:
                self.variants[code].append(article)
        recent = transactions.loc[
            transactions.t_dat.lt(self.as_of)
            & transactions.t_dat.ge(self.as_of - pd.Timedelta(days=28))
        ].sort_values(["customer_id", "t_dat", "article_id"], kind="stable")
        recent = recent.drop_duplicates(["customer_id", "article_id"], keep="last")
        recent = recent.groupby("customer_id", sort=False).tail(4).copy()
        pieces = []
        # All unordered pairs among the last four unique purchases, rather than
        # only adjacent transactions. This is not a reconstructed shopping basket.
        for lag in [1, 2, 3]:
            previous = recent.groupby("customer_id", sort=False).article_id.shift(lag)
            pair = pd.DataFrame({"source": previous, "target": recent.article_id}).dropna()
            pieces.extend([pair, pair.rename(columns={"source": "target", "target": "source"})])
        edges = pd.concat(pieces).groupby(["source", "target"]).size().rename("count").reset_index()
        edges = edges.sort_values(["source", "count", "target"], ascending=[True, False, True])
        edges = edges.groupby("source", sort=False).head(self.co_purchase_neighbours)
        additions = defaultdict(list)
        for row in edges.itertuples(index=False):
            additions[row.source].append((row.target, int(row.count)))
        for article, neighbours in additions.items():
            merged = dict(self.graph.get(article, []))
            for other, count in neighbours:
                merged[other] = max(merged.get(other, 0), count)
            self.graph[article] = list(merged.items())
        return self

    def candidates(self, user):
        ids, matrix = super().candidates(user)
        seen = set(ids)
        extra = []
        affinities = self.affinities.get(user, {})
        categories = sorted(affinities, key=lambda c: (-affinities[c], c))[:self.category_count]
        for category in categories:
            extra.extend(self.category_popular.get(category, []))
        for article, _, _ in self.histories.get(user, [])[:12]:
            extra.extend(self.variants.get(self.product_codes.get(article, article), []))
        extra = [a for a in dict.fromkeys(extra) if a not in seen]
        # Match the original six-feature contract: new articles have no personal
        # repeat signal and no graph score unless already in the base candidates.
        denominator = max(sum(affinities.values()), 1)
        if extra:
            more = np.array([
                [np.log1p(self.count7.get(a, 0)), np.log1p(self.count28.get(a, 0)),
                 0, 0, affinities.get(self.categories.get(a, "Unknown"), 0) / denominator, 0]
                for a in extra
            ], dtype=np.float32)
            ids += extra
            matrix = np.concatenate([matrix, more])
        return ids[:self.max_candidates], matrix[:self.max_candidates]
