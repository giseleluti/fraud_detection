from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
from flask_cors import CORS
import os

from werkzeug.exceptions import BadRequest

app = Flask(__name__)
CORS(app)

current_script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_script_dir, '..', 'modeloperation', 'fraud_detection_model.pkl')
scaler_path = os.path.join(current_script_dir, '..', 'modeloperation', 'scaler.pkl')
model = None
scaler = None

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Modelo e scaler carregados com sucesso!")
except FileNotFoundError:
    print(f"Erro: Arquivos do modelo/scaler não encontrados em {model_path} ou {scaler_path}.")
except Exception as e:
    print(f"Erro ao carregar modelo/scaler: {e}")

EXPECTED_COLUMNS = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]


@app.route('/')
def serve_frontend():
    frontend_dir = os.path.join(current_script_dir, '..', 'frontend')
    return send_from_directory(frontend_dir, 'index.html')


@app.route('/predict_fraud', methods=['POST'])
def predict_fraud():
    try:
        data = request.get_json(force=True)

        if not data:
            return jsonify({"error": "Dados JSON inválidos", "details": "Nenhum dado JSON encontrado na requisição."}), 400

        processed_data = {}
        for col in EXPECTED_COLUMNS:
            if col in data:
                processed_data[col] = data[col]
            else:
                return jsonify({"error": f"Colunas faltando: {col}"}), 400

        input_df = pd.DataFrame([processed_data])

        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[:, 1][0]

        result = {
            "is_fraud": int(prediction),
            "fraud_probability": float(prediction_proba)
        }
        return jsonify(result)

    except BadRequest:
        return jsonify({"error": "Falha ao decodificar JSON", "details": "A requisição não contém JSON válido."}), 400
    except Exception as e:
        return jsonify({"error": "Erro interno do servidor", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)