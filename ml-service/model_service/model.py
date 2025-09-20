# model_service/model.py
import os
import numpy as np
import pandas as pd
from typing import Tuple, List
from joblib import load
from model_service.transform import predict_proba_sklearn
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


def _shap_contrib_sklearn(model, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    # LightGBM sklearn wrapper supports pred_contrib=True
    if hasattr(model, "predict"):
        try:
            contrib = model.predict(X, pred_contrib=True)
            feature_names = list(X.columns)
            return contrib, feature_names
        except Exception:
            pass
    zeros = np.zeros((len(X), len(X.columns)))
    return zeros, list(X.columns)

def _predict_proba_sklearn(model, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict(X)
    if proba.ndim == 2 and proba.shape[1] == 2:
        proba = proba[:, 1]
    print(proba)
    return np.asarray(proba, dtype=float)


def predict_proba_with_shap(model, X: pd.DataFrame):
    # sklearn LGBMClassifier
    proba = predict_proba_sklearn(model, X)
    contrib, feat_names = _shap_contrib_sklearn(model, X)
    return proba, contrib, feat_names
