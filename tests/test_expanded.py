import json

import numpy as np
import pandas as pd

from fashion_recs.data import synthetic_data
from fashion_recs.expanded import ExpandedCandidateEngine
from fashion_recs.model import CandidateEngine
from fashion_recs.experiment import run_comparison
from fashion_recs.pipeline import run


def test_expansion_adds_category_and_product_variants_without_future_items():
    date = pd.Timestamp("2020-01-10")
    articles = pd.DataFrame([
        {"article_id": f"{i:010d}", "product_type_name": "Rare" if i >= 70 else "Common",
         "product_code": "related" if i in [70, 71, 99] else str(i)} for i in range(100)
    ])
    rows = [(date, "others", f"{i:010d}") for i in range(70) for _ in range(3)]
    rows += [(date, "target", "0000000070"), (date, "other", "0000000071")]
    tx = pd.DataFrame(rows, columns=["t_dat", "customer_id", "article_id"])
    cutoff = date + pd.Timedelta(days=1)
    baseline = CandidateEngine().fit(tx, articles, cutoff)
    expanded = ExpandedCandidateEngine(popular_candidates=60).fit(tx, articles, cutoff)
    assert "0000000071" not in baseline.candidates("target")[0]
    assert "0000000071" in expanded.candidates("target")[0]
    assert "0000000099" not in expanded.candidates("target")[0]
    variants_only = ExpandedCandidateEngine(popular_candidates=60, category_count=0).fit(tx, articles, cutoff)
    assert "0000000071" in variants_only.candidates("target")[0]


def test_expanded_features_cannot_see_future_purchases():
    tx, articles, end = synthetic_data()
    cutoff = end - pd.Timedelta(days=7)
    before = ExpandedCandidateEngine().fit(tx, articles, cutoff)
    future = pd.DataFrame([(cutoff, "demo-0000", "9999999999")], columns=tx.columns)
    after = ExpandedCandidateEngine().fit(pd.concat([tx, future]), articles, cutoff)
    ids, matrix = before.candidates("demo-0000")
    ids_after, matrix_after = after.candidates("demo-0000")
    assert ids == ids_after
    np.testing.assert_array_equal(matrix, matrix_after)
    assert len(ids) == len(set(ids)) <= 480
    assert np.isfinite(matrix).all()


def test_rolling_comparison_has_separate_selection_and_confirmation(tmp_path):
    tx, articles, end = synthetic_data()
    report = run_comparison(tx, articles, end, tmp_path, train_users=12, eval_users=12, synthetic=True)
    assert len(report["weeks"]) == 4
    assert [w["role"] for w in report["weeks"]] == ["development"] * 3 + ["previously_seen_confirmation"]
    for week in report["weeks"]:
        assert week["evaluated_users"] == 12
        for variant in week["variants"].values():
            assert variant["feature_max_date"] < week["start"]
            assert variant["training_feature_max_date"] < week["training_labels_start"]
    pairs = [(v, m) for v in ["original", "expanded"] for m in ["hybrid", "learned"]]
    winner = max(pairs, key=lambda p: np.mean([w["variants"][p[0]]["map_at_12"][p[1]] for w in report["weeks"][:-1]]))
    assert (report["selected_variant"], report["selected_method"]) == winner
    assert json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))["data_kind"] == "SYNTHETIC DEMO"


def test_selected_serving_export_matches_hybrid_predictions(tmp_path):
    import pickle
    tx, articles, end = synthetic_data()
    report = run(tx, articles, end, tmp_path, synthetic=True, train_users=20, eval_users=20,
                 engine_class=ExpandedCandidateEngine, serving_method="hybrid", history_days=70)
    assert report["serving_method"] == "hybrid"
    assert report["history_days"] == 70
    with (tmp_path / "model.pkl").open("rb") as stream:
        model = pickle.load(stream)
    demo = json.loads((tmp_path / "demo.json").read_text(encoding="utf-8"))
    assert model["serving_method"] == demo["serving_method"] == "hybrid"
    assert report["engine_class"] == "ExpandedCandidateEngine"
    assert len(model["engine"].recommend("demo-0000", "hybrid")) == 12
