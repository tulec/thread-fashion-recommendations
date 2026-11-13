from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fashion_recs.data import synthetic_data
from fashion_recs.metrics import apk
from fashion_recs.ranking_experiment import Queries, ap_from_matches, review_gate, run_ranking_comparison, score_queries


def test_ranking_is_scoped_to_customer_and_respects_cold_fallback():
    x = np.zeros((5, 6))
    x[:, 0] = [3, 1, 2, 9, 0]
    queries = Queries(x, np.array([0, 1, 1, 0, 1]), np.array([3, 2]),
                      np.array([3, 1]), np.array([False, True]), np.array([0.0, 0.5]))
    pointwise = SimpleNamespace(predict_proba=lambda x: np.c_[x[:, 0], -x[:, 0]])
    lambda_ranker = SimpleNamespace(predict=lambda x, num_threads: -x[:, 0])
    report = score_queries(queries, pointwise, lambda_ranker)
    assert report["map_at_12"]["lambdamart"] == pytest.approx((2 / 3 + 0.5) / 2)
    assert report["map_at_12"]["lambdamart"] == report["oracle_map_at_12"]
    assert report["map_at_12"]["hybrid"] < report["oracle_map_at_12"]
    assert report["cold_users"] == 1
    assert ap_from_matches(np.array([1, 0, 1]), 3) == pytest.approx(apk(["a", "b", "c"], ["a", "x", "b"]))


def test_review_gate_requires_positive_confirmation_interval():
    row = {"paired_lambdamart_minus_hybrid": {"difference": 0.001, "ci_95": [0.0001, 0.002]}}
    uncertain = {"paired_lambdamart_minus_hybrid": {"difference": 0.001, "ci_95": [-0.0001, 0.002]}}
    assert review_gate([row] * 4)["passes"]
    assert not review_gate([row] * 3 + [uncertain])["passes"]


def test_lambdamart_uses_earlier_training_labels_and_same_population(tmp_path):
    pytest.importorskip("lightgbm")
    tx, articles, end = synthetic_data()
    old = tx.assign(t_dat=tx.t_dat - pd.Timedelta(days=91))
    report = run_ranking_comparison(pd.concat([old, tx]), articles, end, tmp_path,
                                    train_users=20, eval_users=20, synthetic=True)
    assert len(report["weeks"]) == 4
    for week in report["weeks"]:
        assert week["evaluated_users"] == week["training_users"] == 20
        assert week["training_feature_max_date"] < week["training_labels_start"] < week["start"]
        assert week["feature_max_date"] < week["start"]
        assert max(week["map_at_12"].values()) <= week["oracle_map_at_12"] + 1e-12
    assert (tmp_path / "ranking.json").exists()
