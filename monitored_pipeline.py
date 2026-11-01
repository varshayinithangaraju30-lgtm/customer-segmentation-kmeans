"""Customer segmentation workflow with logging and monitoring."""

import json
import logging
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
MIN_SILHOUETTE = 0.25  # warn if clustering quality is below this
MAX_SECONDS = 30       # warn if the workflow is slower than this


def get_logger(log_file: str = "segmentation.log") -> logging.Logger:
    """Create a logger that writes to a file and to the screen."""
    logger = logging.getLogger("segmentation")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.addHandler(console)
    return logger


def run_monitored_pipeline(path: str, n_clusters: int = 3,
                           log_file: str = "segmentation.log") -> Tuple[pd.DataFrame, Dict]:
    """Run the full workflow, log every step and return monitoring metrics."""
    logger = get_logger(log_file)
    metrics: Dict = {"status": "failed"}
    start = time.perf_counter()
    logger.info("Workflow started (K=%s)", n_clusters)

    try:
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("Input data is empty")
        metrics["rows_loaded"] = len(df)
        logger.info("Loaded %d rows", len(df))

        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df = df.drop_duplicates().dropna(subset=FEATURES)
        metrics["rows_used"] = len(df)
        dropped = metrics["rows_loaded"] - metrics["rows_used"]
        if dropped:
            logger.warning("Dropped %d rows (duplicates or missing values)", dropped)

        if not isinstance(n_clusters, (int, np.integer)) or n_clusters < 2 or n_clusters >= len(df):
            raise ValueError(f"n_clusters={n_clusters} is invalid for {len(df)} rows")

        X = StandardScaler().fit_transform(df[FEATURES])
        logger.info("Features scaled")

        model = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=42)
        df["Cluster"] = model.fit_predict(X)
        metrics["inertia"] = round(float(model.inertia_), 3)
        metrics["cluster_sizes"] = {int(k): int(v) for k, v in
                                    df["Cluster"].value_counts().sort_index().items()}
        logger.info("K-Means finished, cluster sizes: %s", metrics["cluster_sizes"])

        if min(metrics["cluster_sizes"].values()) == 1:
            logger.warning("A cluster has only 1 customer")

        score = float(silhouette_score(X, df["Cluster"]))
        metrics["silhouette"] = round(score, 3)
        if score < MIN_SILHOUETTE:
            logger.warning("Low silhouette score: %.3f (below %.2f)", score, MIN_SILHOUETTE)
        else:
            logger.info("Silhouette score: %.3f", score)

        metrics["status"] = "success"
        return df, metrics

    except Exception as exc:
        logger.error("Workflow failed: %s", exc)
        metrics["error"] = str(exc)
        raise

    finally:
        metrics["seconds"] = round(time.perf_counter() - start, 3)
        if metrics["seconds"] > MAX_SECONDS:
            logger.warning("Workflow was slow: %.1fs", metrics["seconds"])
        logger.info("Workflow ended: %s", json.dumps(metrics))
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
