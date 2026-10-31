from pathlib import Path

import numpy as np
import pandas as pd


def load_data(directory: Path, lookback_days: int = 98):
    """Read only the recent window, without loading all 31M rows at once."""
    path = directory / "transactions_train.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. See README.md for Kaggle access.")
    latest = None
    total = 0
    for chunk in pd.read_csv(path, usecols=["t_dat"], chunksize=1_000_000):
        maximum = pd.to_datetime(chunk.t_dat).max()
        latest = maximum if latest is None else max(latest, maximum)
        total += len(chunk)
    if latest is None or pd.isna(latest):
        raise ValueError("No transactions found")
    end = latest.normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=lookback_days)
    recent = []
    for chunk in pd.read_csv(
        path, usecols=["t_dat", "customer_id", "article_id"],
        dtype={"customer_id": "string", "article_id": "string"}, chunksize=1_000_000,
    ):
        chunk["t_dat"] = pd.to_datetime(chunk.t_dat)
        recent.append(chunk.loc[chunk.t_dat.ge(start)].copy())
    transactions = pd.concat(recent, ignore_index=True)
    transactions["article_id"] = transactions.article_id.str.zfill(10)
    articles = pd.read_csv(directory / "articles.csv", dtype={"article_id": "string"})
    articles["article_id"] = articles.article_id.str.zfill(10)
    return transactions, articles, end, {
        "source_rows": total, "loaded_rows": len(transactions),
        "lookback_days": lookback_days, "source_max_date": str(latest.date()),
    }


def synthetic_data(seed: int = 42):
    """Small, artificial shopping histories; never presented as H&M results."""
    rng = np.random.default_rng(seed)
    categories = ["Knitwear", "Trousers", "Shirts", "Dresses", "Accessories", "Shoes"]
    palette = ["Black", "Cream", "Blue", "Olive", "Burgundy", "Sand"]
    articles = pd.DataFrame([
        {"article_id": f"{i + 1:010d}", "prod_name": f"{categories[i % 6]} {i // 6 + 1:02d}",
         "product_type_name": categories[i % 6], "colour_group_name": palette[(i // 6) % 6],
         "product_group_name": categories[i % 6]}
        for i in range(180)
    ])
    end = pd.Timestamp("2020-09-23")
    rows = []
    for user in range(260):
        favourite = user % 6
        pool = np.arange(favourite, 180, 6)
        personal = rng.choice(pool, size=7, replace=False)
        for day in range(91):
            if rng.random() > 0.22:
                continue
            for _ in range(int(rng.integers(1, 4))):
                pick = int(rng.choice(personal)) if rng.random() < 0.7 else int(rng.choice(pool))
                if rng.random() < 0.08:
                    pick = int(rng.integers(0, 180))
                rows.append((end - pd.Timedelta(days=91 - day), f"demo-{user:04d}", f"{pick + 1:010d}"))
    return pd.DataFrame(rows, columns=["t_dat", "customer_id", "article_id"]), articles, end

