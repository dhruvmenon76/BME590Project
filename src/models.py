from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def plot_pca_ews(df):
    """
    Performs PCA on 5 EWS features and plots PC1 vs PC2
    colored by class label.
    """

    feature_cols = ["variance", "ac1", "cv", "skewness", "kurtosis"]

    # Extract X and y
    X = df[feature_cols].values
    y = df["label"].values

    # Standardize (VERY IMPORTANT for PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Plot
    plt.figure(figsize=(8,6))

    plt.scatter(
        X_pca[y==0, 0],
        X_pca[y==0, 1],
        alpha=0.6,
        label="Null",
    )

    plt.scatter(
        X_pca[y==1, 0],
        X_pca[y==1, 1],
        alpha=0.6,
        label="Transcritical",
    )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.title("PCA of Early Warning Features")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return pca

    # models.py


from dataclasses import dataclass
from typing import Optional, Dict, Any

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


# -----------------------------
# Individual model constructors
# -----------------------------

def make_logistic_regression(
    C: float = 1.0,
    penalty: str = "l2",
    max_iter: int = 5000,
    class_weight: Optional[str] = None,
    random_state: int = 0,
) -> Pipeline:
    """
    Logistic Regression (scaled).
    """
    clf = LogisticRegression(
        C=C,
        penalty=penalty,
        solver="lbfgs",      # good default for l2
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


def make_gbm(
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    random_state: int = 0,
) -> Pipeline:
    """
    Gradient Boosting Machine (tree-based; scaling not necessary).
    """
    clf = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )
    return Pipeline([
        ("clf", clf),
    ])


def make_knn(
    n_neighbors: int = 15,
    weights: str = "distance",
    metric: str = "minkowski",
    p: int = 2,
) -> Pipeline:
    """
    KNN (scaled).
    """
    clf = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        p=p,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


def make_svm(
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str | float = "scale",
    probability: bool = True,
    class_weight: Optional[str] = None,
    random_state: int = 0,
) -> Pipeline:
    """
    SVM (scaled). probability=True enables predict_proba (slower but useful for AUC).
    """
    clf = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        probability=probability,
        class_weight=class_weight,
        random_state=random_state,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


def make_random_forest(
    n_estimators: int = 500,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 1,
    class_weight: Optional[str] = None,
    random_state: int = 0,
    n_jobs: int = -1,
) -> Pipeline:
    """
    Random Forest (tree-based; scaling not necessary).
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return Pipeline([
        ("clf", clf),
    ])


def make_mlp(
    hidden_layer_sizes: tuple[int, ...] = (32, 16),
    alpha: float = 1e-4,           # L2 weight decay
    learning_rate_init: float = 1e-3,
    max_iter: int = 2000,
    early_stopping: bool = True,
    random_state: int = 0,
) -> Pipeline:
    """
    MLP (scaled). StandardScaler is important for stable training.
    """
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=early_stopping,
        random_state=random_state,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


# ---------------------------------
# Convenience: get all models in one
# ---------------------------------

def get_models(random_state: int = 0) -> Dict[str, Pipeline]:
    """
    Returns a dict of default pipelines keyed by model name.
    """
    return {
        "LogReg": make_logistic_regression(random_state=random_state),
        "GBM": make_gbm(random_state=random_state),
        "KNN": make_knn(),
        "SVM": make_svm(random_state=random_state),
        "RF": make_random_forest(random_state=random_state),
        "MLP": make_mlp(random_state=random_state),
    }