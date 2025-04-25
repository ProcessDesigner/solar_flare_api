from flask import Flask, request, jsonify
from flask_cors import CORS 
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])

# === Load scaler and model ===
scaler = joblib.load("lstm_scaler.pkl")
model = load_model("flare_prediction_model.keras")

# === Config ===
features_count = 33
sequence_length = 24

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # === Case 1: CSV Upload ===
        if "file" in request.files:
            file = request.files["file"]
            df = pd.read_csv(file)

            if df.shape[1] != features_count:
                return jsonify({"error": f"CSV must have exactly {features_count} columns."}), 400

            X = df.values

        # === Case 2: Raw JSON ===
        else:
            data = request.get_json()
            if not data or len(data) < sequence_length or len(data[0]) != features_count:
                return jsonify({
                    "error": f"Invalid input shape. Need at least {sequence_length} rows of {features_count} features each."
                }), 400
            X = np.array(data)

        if len(X) < sequence_length:
            return jsonify({"error": f"Need at least {sequence_length} rows of data."}), 400

        # === Scale input ===
        X_scaled = scaler.transform(X)

        # === Sliding window creation ===
        sequences = []
        for i in range(len(X_scaled) - sequence_length + 1):
            window = X_scaled[i:i+sequence_length]
            sequences.append(window)
        X_seq = np.array(sequences)

        # === Prediction ===
        flare_probs, _ = model.predict(X_seq)
        probs = [float(prob[0]) for prob in flare_probs]

        return jsonify({
            "flare_probabilities": probs,
            "num_sequences": len(probs)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
