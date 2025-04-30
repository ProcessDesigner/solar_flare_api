from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app)

print("🔧 Attempting to load scaler and model...")
scaler = None
model = None

try:
    scaler = joblib.load("lstm_scaler.pkl")
    print("✅ Scaler loaded")
except Exception as e:
    print("❌ Failed to load scaler:", str(e))

try:
    model = load_model("flare_prediction_model.keras")
    print("✅ Model loaded")
except Exception as e:
    print("❌ Failed to load model:", str(e))

# === Config ===
features_count = 33
sequence_length = 24

# 🧮 Utility functions
def calculate_magnetic_field(original_feats):
    fields = ['TOTBSQ', 'USFLUX', 'TOTFX', 'TOTFY', 'TOTFZ']
    return round(sum(original_feats.get(f, 0) for f in fields), 4)

def calculate_intensity_score(original_feats):
    fields = ['MEANPOT', 'TOTPOT', 'R_VALUE', 'SAVNCPP']
    return round(sum(original_feats.get(f, 0) for f in fields), 4)

def classify_flare(prob):
    if prob < 0.2:
        return "No Flare ❄️"
    elif prob < 0.5:
        return "Weak Flare ⚠️"
    elif prob < 0.8:
        return "Moderate Flare 🔥"
    else:
        return "Strong Flare 🌋"

def map_flux_to_flare_class(flux):
    if flux < 1e-7:
        return f"A{flux / 1e-8:.1f}"
    elif flux < 1e-6:
        return f"B{flux / 1e-7:.1f}"
    elif flux < 1e-5:
        return f"C{flux / 1e-6:.1f}"
    elif flux < 1e-4:
        return f"M{flux / 1e-5:.1f}"
    else:
        return f"X{flux / 1e-4:.1f}"

@app.route("/predict", methods=["POST"])
def predict():
    global scaler, model

    if scaler is None or model is None:
        return jsonify({"error": "Model or Scaler not loaded"}), 500

    print("\n📥 Received a prediction request!")

    try:
        # === CSV Upload ===
        if "file" in request.files:
            print("🗂️ File upload detected")
            file = request.files["file"]
            df = pd.read_csv(file)
            if df.shape[1] != features_count:
                return 
                return jsonify({"error": f"CSV must have {features_count} columns."}), 400
            X = df.values
            print(f"✅ CSV shape: {X.shape}")
        else:
            print("📨 Processing raw JSON input")
            data = request.get_json()
            if not data or len(data) < sequence_length or len(data[0]) != features_count:
                return jsonify({
                    "error": f"Invalid input shape. Need ≥ {sequence_length} rows with {features_count} features."
                }), 400
            X = np.array(data)
            print(f"✅ JSON input shape: {X.shape}")

        if len(X) < sequence_length:
            return jsonify({"error": f"At least {sequence_length} rows required."}), 400

        feature_order = scaler.feature_names_in_
        df_input = pd.DataFrame(X, columns=feature_order)[feature_order]

        df_input.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_input.dropna(inplace=True)

        if df_input.shape[0] < sequence_length:
            return jsonify({"error": f"After cleaning, only {df_input.shape[0]} rows remain. Need {sequence_length}."}), 400

        if hasattr(scaler, "scale_") and any(s == 0 for s in scaler.scale_):
            print("⚠️ Constant column detected in scaler — retraining scaler")
            scaler = MinMaxScaler()
            scaler.fit(df_input)

        print("🧪 Scaling input...")
        X_scaled = scaler.transform(df_input)
        if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
            return jsonify({"error": "Scaled input has NaN/Inf."}), 400

        print("📏 Creating sequences...")
        sequences = [X_scaled[i:i+sequence_length] for i in range(len(X_scaled) - sequence_length + 1)]
        X_seq = np.array(sequences)
        print(f"✅ Total sequences: {X_seq.shape[0]}")

        print("🔮 Predicting with model...")
        flare_prob, reconstructed_features = model.predict(X_seq)
        print("🔁 Feature reconstruction shape:", reconstructed_features.shape)

        results = []
        features = list(scaler.feature_names_in_)

        for i in range(len(flare_prob)):
            prob = float(flare_prob[i][0])
            seq_df = pd.DataFrame(X_seq[i], columns=features)

            averaged_scaled = seq_df.mean().values.reshape(1, -1)
            averaged_original = scaler.inverse_transform(averaged_scaled)[0]
            averaged_dict = dict(zip(features, averaged_original.round(4)))
            
            goes_flux = averaged_dict.get("GOES_flux", None)
            if goes_flux:
                flare_class = map_flux_to_flare_class(goes_flux)
            else:
                flare_class = classify_flare(prob)


            result = {
                "sequence_id": i + 1,
                # "flare_probability": round(prob, 6),
                # "flare_class": classify_flare(prob),
                "flare_class": flare_class,
                "magnetic_field_strength": calculate_magnetic_field(averaged_dict),
                "intensity_score": calculate_intensity_score(averaged_dict),
                # "averaged_features": averaged_dict
            }

            # print(f"[{i+1}] Flux: {goes_flux} → Flare Class: {flare_class}")
            results.append(result)

        return jsonify({
            "sequences": results,
            "num_sequences": len(results)
        })

    except Exception as e:
        print("❌ Exception occurred during prediction:")
        print(e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    app.run(host="0.0.0.0", port=8001)
