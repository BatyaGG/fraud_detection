# app.py
import joblib
import os
import traceback
import numpy as np
from typing import List, Dict, Any
from flask import Flask, request, jsonify

from model_service.model import load_model, predict_proba_with_shap
from model_service.transform import to_feature_frame, DEFAULT_FEATURE_ORDER, normalize_input_rows

APP_VERSION = os.getenv("APP_VERSION", "lgbm_iforest_2025-09-16")
FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.90"))

app = Flask(__name__)
# model = load_model("weights/lightgbm.bin")

loaded_data = joblib.load('prod_data.joblib')
model = loaded_data['model']
features = loaded_data['features']
shap_values = loaded_data['shap_values']


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

        # keep original order; normalize -> features
        rows_norm = normalize_input_rows(rows)
        X, row_ids = to_feature_frame(rows_norm)

        proba, shap_contribs, feature_names = predict_proba_with_shap(model, X)

        scores = []
        for i in range(len(X)):
            contrib = shap_contribs[i]
            # LightGBM pred_contrib includes bias as last column; drop it if present
            feat_vals = contrib[:-1] if len(contrib) == len(feature_names) + 1 else contrib

            top_idx = np.argsort(-np.abs(feat_vals))[:3]
            top_features = [[feature_names[j], float(round(float(feat_vals[j]), 4))] for j in top_idx]

            ml_score = float(round(float(proba[i]), 6))
            anomaly_score = ml_score  # simple for now; replace with IForest later

            hints = []
            if "log_amount" in X.columns and float(X.iloc[i]["log_amount"]) > 10:
                hints.append("high amount")
            if "delta_origin" in X.columns and float(X.iloc[i]["delta_origin"]) < 0:
                hints.append("origin balance drop")
            if "type_PAYMENT" in X.columns and int(X.iloc[i]["type_PAYMENT"]) == 1:
                hints.append("payment pattern")

            # ---- MOCK FRAUD OVERRIDE ----
            # If the original input row contains the parameter "mock_fraud",
            # force this row to be treated as fraud.
            if i < len(rows) and ("mock_fraud" in rows[i]):
                ml_score = max(ml_score, 0.99)
                anomaly_score = max(anomaly_score, 0.99)
                # mark the reason and make sure top_features includes a visible flag
                hints.insert(0, "MOCK_FRAUD override")
                top_features = [["mock_flag", 0.99]] + top_features[:2]

            reason_hint = "; ".join(hints) if hints else "pattern check requested"

            scores.append({
                "row_id": int(row_ids[i]),
                "ml_score": ml_score,
                "anomaly_score": float(round(anomaly_score, 6)),
                "top_features": top_features,
                "reason_hint": reason_hint
            })

        # Collect exact input rows flagged as fraud (by threshold)
        fraud_indices = [i for i, s in enumerate(scores) if s["ml_score"] >= FRAUD_THRESHOLD]
        fraud_rows = []
        for i in fraud_indices:
            enriched = dict(rows[i])  # original row
            enriched.update({
                "row_index": i,
                "row_id": scores[i]["row_id"],
                "ml_score": scores[i]["ml_score"],
                "anomaly_score": scores[i]["anomaly_score"],
                "reason_hint": scores[i]["reason_hint"],
                "top_features": scores[i]["top_features"],
            })
            fraud_rows.append(enriched)

        return jsonify({
            "model_version": APP_VERSION,
            "fraud_threshold": FRAUD_THRESHOLD,
            "fraud_count": len(fraud_rows),
            "fraud_row_indices": fraud_indices,
            "fraud_rows": fraud_rows,
            "scores": scores,
            "warnings": []
        }), 200

    except Exception as e:
        app.logger.error("Scoring error: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
