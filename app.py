from flask import Flask, request, jsonify
import torch
import numpy as np
from model import LSTMModel

app = Flask(__name__)

# === Paramètres du modèle ===
input_size = 18         
hidden_size = 64        
model_path = "lstm_trained.pth"

# === Chargement du modèle ===
model = LSTMModel(input_size=input_size, hidden_size=hidden_size)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# === Endpoint principal ===
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if "sequence" not in data:
            return jsonify({"error": "Requête invalide : champ 'sequence' manquant."}), 400

        sequence = data["sequence"]

        # Vérification de la forme (liste de listes)
        if not isinstance(sequence, list) or not all(isinstance(row, list) for row in sequence):
            return jsonify({"error": "Le champ 'sequence' doit être une liste de listes."}), 400

        # Convertir en tenseur (1 batch)
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # shape: (1, seq_len, features)

        with torch.no_grad():
            prob = model(x).item()
            prediction = int(prob > 0.5)

        return jsonify({
            "prediction": prediction,
            "probability": round(prob, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# === Lancer l'API ===
if __name__ == "__main__":
    app.run(debug=True)