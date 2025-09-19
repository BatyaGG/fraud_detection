# model_service/transform.py
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

DEFAULT_FEATURE_ORDER = ['oldbalanceOrg', 'oldbalanceOrg_7_sum', 'balanceDelta_7_sum', 'balanceChngOrig',
                         'balanceChngOrig_7_sum', 'balanceDelta', 'balanceChngOrig_7_avg', 'oldbalanceOrg_14_avg',
                         'balanceChngOrig_14_avg', 'balanceChngOrig_28_sum', 'balanceChngOrig_14_sum',
                         'balanceDelta_14_sum', 'oldbalanceOrg_90_sum', 'balanceDelta_28_sum', 'balanceDelta_7_avg',
                         'balanceDelta_28_avg', 'balanceDelta_14_avg', 'oldbalanceOrg_14_sum', 'oldbalanceOrg_7_avg',
                         'oldbalanceOrg_28_avg', 'balanceDelta_180_avg', 'oldbalanceOrg_90_avg', 'balanceDelta_90_avg',
                         'oldbalanceOrg_28_sum', 'balanceDelta_90_sum', 'balanceChngOrig_180_avg', 'amount',
                         'balanceChngOrig_90_avg', 'balanceChngOrig_90_sum', 'balanceDelta_180_sum']


def _parse_date(d):
    if d in (None, "", "null"):
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d.%m.%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(str(d), fmt)
        except Exception:
            continue
    return None


def _norm_vendor(name: str) -> str:
    if name is None:
        return ""
    s = re.sub(r"[^A-Z0-9 ]", "", str(name).upper()).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _to_float(x: Any) -> float:
    """Robust numeric coercion (handles None, ints/floats, and '1,234.56' strings)."""
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(',', '')
    try:
        return float(s)
    except ValueError:
        return 0.0


def normalize_input_rows(rows: List[Dict[str, Any]], features) -> List[Dict[str, Any]]:
    """
    Convert Purchase-Order-like rows into model feature rows.
    - vouchers_paid_sum -> balanceChngOrig
    - Encumbered Amount -> amount
    - all other model features -> 0.0
    Returns a list of dicts with keys in MODEL_FEATURES order.
    """
    out: List[Dict[str, Any]] = []

    for r in rows:
        feat = {k: 0.0 for k in features}  # start with zeroes
        feat['balanceChngOrig'] = _to_float(r.get('vouchers_paid_sum', 0))
        feat['amount'] = _to_float(r.get('Encumbered Amount', 0))
        out.append(feat)

    return out


def to_feature_frame(rows_norm: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Маппит «закупочный» JSON → «paySim-like» → фичи для модели.
    Возвращает (X, row_ids) в строго заданном порядке колонок DEFAULT_FEATURE_ORDER.
    """
    df = pd.DataFrame(rows_norm)
    df["row_id"] = range(len(df))

    X = df[DEFAULT_FEATURE_ORDER].copy()
    row_ids = df["row_id"].values
    return X, row_ids
