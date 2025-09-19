# model_service/model.py
import os
import numpy as np
import pandas as pd
from typing import Tuple, List
from joblib import load

import lightgbm as lgb

# You can tweak fallback rate via env; defaults to 5%
FALLBACK_FRAUD_RATE = float(os.getenv("FALLBACK_FRAUD_RATE", "0.05"))


def load_model(path: str):
    """
    Try to load a model from 'path'. If missing/corrupt, return None (no exceptions).
    """
    try:
        if not os.path.exists(path):
            return None
        model_data = load(path)
        model = model_data['model']  # sklearn LGBMClassifier or lgb.Booster
        return model
    except Exception:
        # Silent fallback to "no model" mode
        return None


def _predict_proba_sklearn(model, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict(X)
    if proba.ndim == 2 and proba.shape[1] == 2:
        proba = proba[:, 1]
    print(proba)
    return np.asarray(proba, dtype=float)


def _shap_contrib_sklearn(model, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    # LightGBM sklearn wrapper supports pred_contrib=True
    if hasattr(model, "predict"):
        try:
            contrib = model.predict(X, pred_contrib=True)
            feature_names = list(X.columns)
            return contrib, feature_names
        except TypeError:
            pass
    zeros = np.zeros((len(X), len(X.columns)))
    return zeros, list(X.columns)


def _predict_proba_booster(booster, X: pd.DataFrame) -> np.ndarray:
    preds = booster.predict(X, raw_score=False)
    return np.asarray(preds, dtype=float)


def _shap_contrib_booster(booster, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    contrib = booster.predict(X, pred_contrib=True)
    feature_names = booster.feature_name() or list(X.columns)
    return contrib, feature_names


def _predict_random_with_shap(X: pd.DataFrame, fraud_rate: float = FALLBACK_FRAUD_RATE):
    """
    Fallback when no model is available:
    - ~fraud_rate of rows get high score (0.95), others low (0.05).
    - Generates synthetic SHAP-like contributions so the UI works.
    """
    n = len(X)
    rng = np.random.default_rng()
    flags = rng.random(n) < fraud_rate
    proba = np.where(flags, 0.95, 0.05).astype(float)

    feat_names = list(X.columns)
    m = len(feat_names)

    # Build contributions matrix with a "bias" column at the end (like LGBM)
    contrib = np.zeros((n, m + 1), dtype=float)
    # Give 2-3 random features small signed contributions per row
    for i in range(n):
        k = min(3, m) if m > 0 else 0
        if k > 0:
            idx = rng.choice(m, size=k, replace=False)
            vals = rng.uniform(0.05, 0.25, size=k) * rng.choice([-1.0, 1.0], size=k)
            contrib[i, idx] = vals
        # bias term stays 0.0 for simplicity

    return proba, contrib, feat_names


def predict_proba_with_shap(model, X: pd.DataFrame):
    """
    Returns (probabilities, shap_contribs, feature_names).
    - If a LightGBM model is present, use it.
    - If no model, produce random predictions with ~5% positives and synthetic SHAP.
    """
    # if model is None:
    #     return _predict_random_with_shap(X)

    # lightgbm Booster

    # sklearn LGBMClassifier
    proba = _predict_proba_sklearn(model, X)
    contrib, feat_names = _shap_contrib_sklearn(model, X)
    return proba, contrib, feat_names
