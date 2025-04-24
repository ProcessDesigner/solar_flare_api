from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# === Load scaler and model ===
scaler = joblib.load("lstm_scaler.pkl")
model = load_model("final_flare_model.keras")

# === Config ===
features_count = 33
sequence_length = 24

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Check minimum length
    if not data or len(data) < sequence_length or len(data[0]) != features_count:
        return jsonify({
            "error": f"Invalid input shape. Need at least {sequence_length} rows of 33 features each."
        }), 400

    try:
        X = np.array(data)  # shape: (N, 33)
        X_scaled = scaler.transform(X)

        # === Sliding window creation ===
        sequences = []
        for i in range(len(X_scaled) - sequence_length + 1):
            window = X_scaled[i:i+sequence_length]
            sequences.append(window)
        X_seq = np.array(sequences)  # shape: (num_sequences, 24, 33)

        # === Model Prediction ===
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
