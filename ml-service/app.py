# app.py
import os
import traceback
import numpy as np
from typing import List, Dict, Any
from flask import Flask, request, jsonify

from model_service.model import load_model, predict_proba_with_shap
from model_service.transform import to_feature_frame, DEFAULT_FEATURE_ORDER, normalize_input_rows

APP_VERSION = os.getenv("APP_VERSION", "lgbm_iforest_2025-09-16")

app = Flask(__name__)
model = load_model("weights/lightgbm.bin")  # if missing, model=None and we fallback


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/version", methods=["GET"])
def version():
    return jsonify({"model_version": APP_VERSION}), 200


@app.route("/score", methods=["POST"])
def score():
    try:
        payload = request.get_json(force=True, silent=False)
        rows: List[Dict[str, Any]] = payload.get("rows", [])
        if not isinstance(rows, list) or len(rows) == 0:
            return jsonify({"error": "payload must contain non-empty 'rows' list"}), 400

        rows_norm = normalize_input_rows(rows)
        X, row_ids = to_feature_frame(rows_norm)

        proba, shap_contribs, feature_names = predict_proba_with_shap(model, X)

        scores = []
        for i in range(len(X)):
            contrib = shap_contribs[i]
            if len(contrib) == len(feature_names) + 1:
                feat_vals = contrib[:-1]
            else:
                feat_vals = contrib

            top_idx = np.argsort(-np.abs(feat_vals))[:3]
            top_features = [[feature_names[j], float(round(feat_vals[j], 4))] for j in top_idx]

            ml_score = float(round(float(proba[i]), 6))
            anomaly_score = ml_score  # keep simple; replace with IForest later

            hints = []
            if "log_amount" in X.columns and X.iloc[i]["log_amount"] > 10:
                hints.append("high amount")
            if "delta_origin" in X.columns and X.iloc[i]["delta_origin"] < 0:
                hints.append("origin balance drop")
            if "type_PAYMENT" in X.columns and X.iloc[i]["type_PAYMENT"] == 1:
                hints.append("payment pattern")
            reason_hint = "; ".join(hints) if hints else "pattern check requested"

            scores.append({
                "row_id": int(row_ids[i]),
                "ml_score": ml_score,
                "anomaly_score": float(round(anomaly_score, 6)),
                "top_features": top_features,
                "reason_hint": reason_hint
            })

        return jsonify({
            "model_version": APP_VERSION,
            "scores": scores,
            "warnings": []
        }), 200

    except Exception as e:
        app.logger.error("Scoring error: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
