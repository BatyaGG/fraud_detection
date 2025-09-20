# app.py
import joblib
import os
import traceback
import numpy as np
import uuid
import re
from pathlib import Path
from typing import List, Dict, Any
from flask import Flask, jsonify, request, send_file, abort, url_for

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

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

UUID_HEX_RE = re.compile(r"^[0-9a-f]{32}$", re.IGNORECASE)


def save_scores_csv(scores: List[float]) -> str:
    """
    Save scores to a single-column CSV (header 'score') under a random UUID.
    Returns the UUID hex string (without hyphens).
    """
    file_id = uuid.uuid4().hex
    csv_path = RESULTS_DIR / f"{file_id}.csv"

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("score\n")
        for s in scores:
            f.write(f"{float(s)}\n")

    return file_id


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

        scores, contrib, feat_names = predict_proba_with_shap(model, X)
        fraud_indices = [i for i, score in enumerate(scores) if score >= FRAUD_THRESHOLD]
        fraud_rows = []
        fraud_scores = []
        for idx in fraud_indices:
            row_norm = rows_norm[idx]
            row = {feat: row_norm.get(feat, 0) for feat in features}
            fraud_scores.append(scores[idx])
            fraud_rows.append(row)

        file_id = save_scores_csv(scores)
        download_url = url_for("download_score", file_id=file_id, _external=True)

        return jsonify({
            "model_version": APP_VERSION,
            "fraud_threshold": FRAUD_THRESHOLD,
            "fraud_count": len(fraud_rows),
            "fraud_row_indices": fraud_indices,
            "ml_scores": fraud_scores,
            "fraud_rows": fraud_rows,
            "scores_download_url": download_url
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


@app.route("/results/<file_id>", methods=["GET"])
def download_score(file_id: str):
    """Download endpoint: serves the CSV by its UUID."""
    if not UUID_HEX_RE.fullmatch(file_id):
        abort(400, description="invalid file id")

    csv_path = RESULTS_DIR / f"{file_id}.csv"
    if not csv_path.exists():
        abort(404, description="file not found")

    return send_file(
        csv_path,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"{file_id}.csv",
        max_age=0,
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
