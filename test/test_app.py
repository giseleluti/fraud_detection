import pytest
import os
import sys
import joblib
import pandas as pd
from unittest.mock import patch

current_test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_test_dir, '..')
backend_dir = os.path.join(project_root, 'backend')
sys.path.insert(0, backend_dir)


class DummyModel:
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            return [1] if X["Amount"].iloc[0] > 1000 else [0]
        else:
            return [1] if X[0, EXPECTED_COLUMNS.index("Amount")] > 1000 else [0]


class DummyScaler:
    def transform(self, X):
        return X.values


from backend.app import app, EXPECTED_COLUMNS, model, scaler


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path):
    model_ops_dir = tmp_path / "model-operation"
    model_ops_dir.mkdir()

    joblib.dump(DummyModel(), model_ops_dir / "fraud_detection_model.pkl")
    joblib.dump(DummyScaler(), model_ops_dir / "scaler.pkl")

    with patch('app.joblib.load') as mock_joblib_load:
        mock_joblib_load.side_effect = [DummyModel(), DummyScaler()]
        global model, scaler
        model = DummyModel()
        scaler = DummyScaler()
        yield


def test_serve_frontend(client, setup_test_environment):
    res = client.get('/')
    assert res.status_code == 200
    assert b"Detector de Fraude Financeira" in res.data


def test_predict_fraud_missing_column(client, setup_test_environment):
    test_data = {col: i * 0.1 for i, col in enumerate(EXPECTED_COLUMNS)}
    del test_data['Time']
    res = client.post('/predict_fraud', json=test_data)
    assert res.status_code == 400
    data = res.get_json()
    assert "Colunas faltando: " 'Time' in data['error']


def test_predict_fraud_invalid_json(client, setup_test_environment):
    res = client.post('/predict_fraud', data="this is not json", content_type='text/plain')
    assert res.status_code == 400
    data = res.get_json()
    assert 'error' in data
    assert "A requisição não contém JSON válido." in data['details']

