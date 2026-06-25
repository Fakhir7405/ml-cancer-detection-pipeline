from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load everything
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('features.pkl', 'rb') as f:
    feature_names = pickle.load(f)

print(f"✅ Model loaded. Expects {len(feature_names)} features.")

@app.route('/')
def home():
    return jsonify({
        "message": "Cancer Detection ML API",
        "model": "Random Forest / Logistic Regression",
        "features_required": len(feature_names),
        "feature_names": feature_names[:5],
        "usage": "POST to /predict with feature values"
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'features' not in data:
        return jsonify({
            "error": "Send features as a list",
            "example": {"features": [17.99, 10.38, 122.8, 1001.0, 0.1184]}
        }), 400

    features = data['features']

    if len(features) != len(feature_names):
        return jsonify({
            "error": f"Expected {len(feature_names)} features, got {len(features)}"
        }), 400

    # Scale and predict
    features_array = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]

    return jsonify({
        "prediction": "Benign" if prediction == 1 else "Malignant",
        "confidence": f"{probability.max() * 100:.1f}%",
        "probabilities": {
            "malignant": f"{probability[0] * 100:.1f}%",
            "benign": f"{probability[1] * 100:.1f}%"
        }
    })

if __name__ == '__main__':
    app.run(debug=True)