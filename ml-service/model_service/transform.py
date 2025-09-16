# model_service/transform.py
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

# Желаемый порядок фич (должен совпадать с обучением модели)
DEFAULT_FEATURE_ORDER = [
    "step",
    "amount",
    "log_amount",
    "delta_origin",
    "delta_dest",
    "type_CASH_OUT",
    "type_CASH_IN",
    "type_TRANSFER",
    "type_PAYMENT",
    "type_DEBIT",
]

TX_TYPES = {"CASH_OUT", "CASH_IN", "TRANSFER", "PAYMENT", "DEBIT"}


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


def normalize_input_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Приводим вход к единому виду (числа/даты/строки).
    Ожидаем "закупочный" JSON; при отсутствии полей выставляем дефолты.
    """
    normed = []
    for i, r in enumerate(rows):
        amount = _to_float(r.get("amount"))
        qty = _to_float(r.get("qty", 1.0))
        unit_price = _to_float(r.get("unit_price", None))
        if unit_price is None and amount is not None and qty not in (None, 0):
            unit_price = amount / max(1.0, qty)

        po_date = _parse_date(r.get("po_date"))
        inv_date = _parse_date(r.get("invoice_date"))
        pay_date = _parse_date(r.get("payment_date"))

        # step — аналог "времени шага" (дней от PO до invoice или до payment)
        step = 0
        if po_date and inv_date:
            step = max(0, (inv_date - po_date).days)
        elif inv_date and pay_date:
            step = max(0, (pay_date - inv_date).days)

        tx_type = str(r.get("type") or r.get("tx_type") or "PAYMENT").upper()
        if tx_type not in TX_TYPES:
            tx_type = "PAYMENT"

        # Для "псевдо-paySim" полей (балансы) сделаем простую модель:
        oldbalanceOrg = _to_float(r.get("oldbalanceOrg", amount or 0.0))
        newbalanceOrg = _to_float(r.get("newbalanceOrig", (oldbalanceOrg or 0.0) - (amount or 0.0)))
        oldbalanceDest = _to_float(r.get("oldbalanceDest", 0.0))
        newbalanceDest = _to_float(r.get("newbalanceDest", (oldbalanceDest or 0.0) + (amount or 0.0)))

        normed.append({
            "row_id": r.get("row_id", i),
            "vendor": _norm_vendor(r.get("vendor") or r.get("vendor_name")),
            "po_number": r.get("po_number"),
            "invoice_number": r.get("invoice_number"),
            # псевдо-paySim поля
            "step": int(step),
            "type": tx_type,
            "amount": float(amount or 0.0),
            "nameOrig": _norm_vendor(r.get("buyer") or "BUYER"),
            "oldbalanceOrg": float(oldbalanceOrg or 0.0),
            "newbalanceOrig": float(newbalanceOrg or 0.0),
            "nameDest": _norm_vendor(r.get("vendor") or "VENDOR"),
            "oldbalanceDest": float(oldbalanceDest or 0.0),
            "newbalanceDest": float(newbalanceDest or 0.0),
        })
    return normed


def _to_float(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def to_feature_frame(rows_norm: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Маппит «закупочный» JSON → «paySim-like» → фичи для модели.
    Возвращает (X, row_ids) в строго заданном порядке колонок DEFAULT_FEATURE_ORDER.
    """
    df = pd.DataFrame(rows_norm)

    # Базовые числовые фичи
    df["log_amount"] = (df["amount"] + 1e-9).apply(np.log)
    df["delta_origin"] = (df["newbalanceOrig"] - df["oldbalanceOrg"]).astype(float)
    df["delta_dest"] = (df["newbalanceDest"] - df["oldbalanceDest"]).astype(float)

    # One-hot для типа транзакции
    for t in ["CASH_OUT", "CASH_IN", "TRANSFER", "PAYMENT", "DEBIT"]:
        col = f"type_{t}"
        df[col] = (df["type"] == t).astype(int)

    # Гарантируем порядок фич; отсутствующие заполним нулями
    for col in DEFAULT_FEATURE_ORDER:
        if col not in df.columns:
            df[col] = 0

    X = df[DEFAULT_FEATURE_ORDER].copy()
    row_ids = df["row_id"].values
    return X, row_ids
