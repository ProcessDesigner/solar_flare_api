from flask import Flask, request, jsonify
from flask_cors import CORS 
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# === Load scaler and model ===
print("🔧 Loading scaler and model...")
scaler = joblib.load("lstm_scaler.pkl")
model = load_model("flare_prediction_model.keras")
print("✅ Model and scaler loaded successfully!")

# === Config ===
features_count = 33
sequence_length = 24

@app.route("/predict", methods=["POST"])
def predict():
    print("\n📥 Received a prediction request!")

    try:
        # === Case 1: CSV Upload ===
        if "file" in request.files:
            print("🗂️ Detected file upload")
            file = request.files["file"]
            df = pd.read_csv(file)

            if df.shape[1] != features_count:
                return jsonify({"error": f"CSV must have exactly {features_count} columns."}), 400

            X = df.values
            print(f"✅ Loaded CSV with shape: {X.shape}")

        # === Case 2: Raw JSON ===
        else:
            print("📨 Processing raw JSON input")
            data = request.get_json()
            if not data or len(data) < sequence_length or len(data[0]) != features_count:
                return jsonify({
                    "error": f"Invalid input shape. Need at least {sequence_length} rows of {features_count} features each."
                }), 400
            X = np.array(data)
            print(f"✅ Parsed JSON with shape: {X.shape}")

        # === Check sequence length ===
        if len(X) < sequence_length:
            return jsonify({"error": f"Need at least {sequence_length} rows of data."}), 400

        print("🧪 Scaling input...")
        X_scaled = scaler.transform(X)

        print("📊 Creating sequences...")
        sequences = []
        for i in range(len(X_scaled) - sequence_length + 1):
            window = X_scaled[i:i+sequence_length]
            sequences.append(window)
        X_seq = np.array(sequences)
        print(f"✅ Created {len(X_seq)} sequence(s) with shape: {X_seq.shape}")

        print(" Running model prediction...")
        flare_probs, _ = model.predict(X_seq)
        probs = [float(prob[0]) for prob in flare_probs]
        print(f"✅ Prediction complete! Sample: {probs[:3]}")

        return jsonify({
            "flare_probabilities": probs,
            "num_sequences": len(probs)
        })

    except Exception as e:
        print(" Exception occurred during prediction:")
        print(e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=8000)
