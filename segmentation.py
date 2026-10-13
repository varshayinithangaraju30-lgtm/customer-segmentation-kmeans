"""Reusable K-Means customer segmentation functions."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]


def load_data(path):
    """Load the dataset and make sure it is not empty."""
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Input data is empty")
    return df


def preprocess(df, features=FEATURES):
    """Remove duplicates and missing values, then scale the features."""
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.copy().drop_duplicates().dropna(subset=features)
    X = StandardScaler().fit_transform(df[features])
    return df, X


def validate(X, n_clusters):
    """Check that the data and the number of clusters are valid."""
    if np.isnan(X).any() or not np.isfinite(X).all():
        raise ValueError("Data has NaN or infinite values")
    if n_clusters < 1 or n_clusters > X.shape[0]:
        raise ValueError("n_clusters must be between 1 and number of rows")


def cluster(X, n_clusters=3):
    """Run K-Means with K-Means++ initialization."""
    model = KMeans(n_clusters=n_clusters, init="k-means++",
                   n_init=10, random_state=42)
    return model, model.fit_predict(X)


def evaluate(X, labels):
    """Return silhouette score and Calinski-Harabasz index."""
    if len(set(labels)) < 2:
        raise ValueError("Need at least 2 clusters to evaluate")
    return {
        "silhouette": silhouette_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
    }


def reduce_to_2d(X):
    """Reduce features to 2 dimensions using PCA."""
    if X.shape[1] < 2:
        raise ValueError("Need at least 2 features for PCA")
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X), pca


def run_pipeline(path, n_clusters=3):
    """Load, preprocess, validate and cluster the data in one call."""
    df = load_data(path)
    df, X = preprocess(df)
    validate(X, n_clusters)
    model, labels = cluster(X, n_clusters)
    df["Cluster"] = labels
    return df, X, model
