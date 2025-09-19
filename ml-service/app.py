# app.py
import joblib
import os
import traceback
import numpy as np
from typing import List, Dict, Any
from flask import Flask, request, jsonify

from model_service.model import predict_proba_with_shap
from model_service.transform import to_feature_frame, normalize_input_rows

APP_VERSION = os.getenv("APP_VERSION", "lgbm_iforest_2025-09-16")
FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.90"))

app = Flask(__name__)

loaded_data = joblib.load('weights/prod_data.joblib')
model = loaded_data['model']
features = loaded_data['features']

shap_values = loaded_data['shap_values']
shap_values = shap_values[np.random.permutation(shap_values.shape[0])[:100]]
shap_values = np.round(shap_values, 3)


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
        rows_norm = normalize_input_rows(rows, features)
        X, row_ids = to_feature_frame(rows_norm)

        proba, shap_contribs, feature_names = predict_proba_with_shap(model, X)

        scores = []
        fraud_indices = []
        fraud_rows = []
        for i in range(len(X)):
            contrib = shap_contribs[i]
            # LightGBM pred_contrib includes bias as last column; drop it if present
            feat_vals = contrib[:-1] if len(contrib) == len(feature_names) + 1 else contrib

            top_idx = np.argsort(-np.abs(feat_vals))[:3]
            top_features = [{feature_names[j]: float(round(float(feat_vals[j]), 4))} for j in top_idx]

            ml_score = float(round(float(proba[i]), 6))
            anomaly_score = ml_score  # simple for now; replace with IForest later

            score_dict = {
                "row_id": int(row_ids[i]),
                "ml_score": ml_score,
                "anomaly_score": float(round(anomaly_score, 6)),
            }
            if score_dict["ml_score"] >= FRAUD_THRESHOLD:
                enriched = dict(rows[i])  # original row
                enriched.update({
                    "row_index": i,
                    "row_id": scores[i]["row_id"],
                    "ml_score": scores[i]["ml_score"],
                    "reason_hint": "Fraud detected by model",
                    "top_features": top_features,
                })
                fraud_rows.append(enriched)

            scores.append(score_dict)

        # Collect exact input rows flagged as fraud (by threshold)
        fraud_indices = [i for i, s in enumerate(scores) if s["ml_score"] >= FRAUD_THRESHOLD]
        fraud_rows = []

        return jsonify({
            "model_version": APP_VERSION,
            "fraud_threshold": FRAUD_THRESHOLD,
            "fraud_count": len(fraud_rows),
            "fraud_row_indices": fraud_indices,
            "fraud_rows": fraud_rows,
            "scores": scores,
            "warnings": [],
            'features': features,
        }), 200

    except Exception as e:
        app.logger.error("Scoring error: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/shap", methods=["GET"])
def shap():
    try:
        return jsonify({
            "shap": shap_values.tolist()
        }), 200

    except Exception as e:
        app.logger.error("Scoring error: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
