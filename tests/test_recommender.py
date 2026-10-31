import json

import numpy as np
import pandas as pd
import pytest

from fashion_recs.data import synthetic_data, load_data
from fashion_recs.metrics import apk
from fashion_recs.model import CandidateEngine
from fashion_recs.pipeline import run, submit


def test_ap_handles_duplicates_and_denominator():
    assert apk(["a", "b"], ["a", "a", "b"]) == pytest.approx((1 + 2 / 3) / 2)
    assert apk(["a", "b", "c"], ["a"], k=2) == 0.5
    assert apk([], ["a"]) == 0
    assert apk(["a"], ["b"]) == 0


def test_future_purchase_cannot_affect_candidates_or_features():
    tx, articles, end = synthetic_data()
    cutoff = end - pd.Timedelta(days=7)
    before = CandidateEngine().fit(tx, articles, cutoff)
    injected = pd.DataFrame([
        (cutoff, "demo-0000", "9999999999"),
        (cutoff + pd.Timedelta(days=5), "demo-0000", "9999999999"),
    ], columns=tx.columns)
    after = CandidateEngine().fit(pd.concat([tx, injected]), articles, cutoff)
    ids1, x1 = before.candidates("demo-0000")
    ids2, x2 = after.candidates("demo-0000")
    assert before.max_seen_date < cutoff
    assert ids1 == ids2
    np.testing.assert_array_equal(x1, x2)
    assert "9999999999" not in ids2


def test_cold_start_and_unique_recommendations():
    tx, articles, end = synthetic_data()
    engine = CandidateEngine().fit(tx, articles, end)
    assert engine.recommend("unseen") == engine.recommend("unseen", "popular")
    for user in ["unseen", "demo-0000", "demo-0001"]:
        recs = engine.recommend(user)
        assert len(recs) == len(set(recs)) == 12
        assert all(len(a) == 10 for a in recs)


def test_csv_loading_preserves_article_zeroes(tmp_path):
    tx, articles, end = synthetic_data()
    tx.to_csv(tmp_path / "transactions_train.csv", index=False)
    articles.to_csv(tmp_path / "articles.csv", index=False)
    loaded, catalogue, actual_end, provenance = load_data(tmp_path)
    assert actual_end == end
    assert loaded.article_id.str.len().eq(10).all()
    assert catalogue.article_id.str.len().eq(10).all()
    assert provenance["source_rows"] == len(tx)


def test_end_to_end_artifacts_and_synthetic_submission_guard(tmp_path):
    tx, articles, end = synthetic_data()
    report = run(tx, articles, end, tmp_path, synthetic=True, train_users=30, eval_users=30)
    assert report["evaluated_users"] == 30
    assert all(0 <= value <= 1 for value in report["map_at_12"].values())
    assert pd.Timestamp(report["evaluation_feature_max_date"]) < pd.Timestamp(report["evaluation_labels_start"])
    demo = json.loads((tmp_path / "demo.json").read_text(encoding="utf-8"))
    assert demo["data_kind"].startswith("SYNTHETIC")
    assert all(len(u["recommendations"]) == 12 for u in demo["users"])
    assert all(u["id"].startswith("Покупатель") for u in demo["users"])
    with pytest.raises(ValueError, match="Synthetic"):
        submit(tmp_path / "model.pkl", tmp_path / "nonexistent.csv", tmp_path / "submission.csv")

