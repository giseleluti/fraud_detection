import pandas as pd
import os  # Importar para manipulação de caminhos
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, \
    f1_score  # Importar mais métricas
import joblib


def train_and_save_model(data_path, model_output_dir):
    """
    Treina o modelo de detecção de fraudes e salva o modelo e o scaler.

    Args:
        data_path (str): Caminho completo para o arquivo CSV do dataset.
        model_output_dir (str): Diretório para salvar o modelo e o scaler.
    """
    try:
        df = pd.read_csv(data_path)
        print(f"Dataset carregado com sucesso de: {data_path}")
        print(f"Formato do dataset: {df.shape}")
    except FileNotFoundError:
        print(f"Erro: O arquivo do dataset não foi encontrado em '{data_path}'.")
        return
    except Exception as e:
        print(f"Erro ao carregar o dataset: {e}")
        return

    # Separar features (X) e target (y)
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Escalar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # Divisão em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42, stratify=y)

    # Lidar com desbalanceamento (SMOTE no conjunto de treino)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Treinar o modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    print("\nIniciando treinamento do modelo RandomForest...")
    model.fit(X_train_res, y_train_res)
    print("Treinamento concluído.")

    # Avaliar o modelo
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n--- Relatório de Classificação no conjunto de Teste ---")
    print(classification_report(y_test, y_pred, target_names=['Não Fraude', 'Fraude']))

    auc_score = roc_auc_score(y_test, y_prob)
    precision_fraud = precision_score(y_test, y_pred, pos_label=1)
    recall_fraud = recall_score(y_test, y_pred, pos_label=1)
    f1_fraud = f1_score(y_test, y_pred, pos_label=1)

    print(f"AUC-ROC: {auc_score:.4f}")
    print(f"Precisão (Fraude): {precision_fraud:.4f}")
    print(f"Recall (Fraude): {recall_fraud:.4f}")
    print(f"F1-Score (Fraude): {f1_fraud:.4f}")

    # Salvar o modelo treinado e o scaler
    os.makedirs(model_output_dir, exist_ok=True)  # Garante que a pasta exista
    model_path = os.path.join(model_output_dir, 'fraud_detection_model.pkl')
    scaler_path = os.path.join(model_output_dir, 'scaler.pkl')

    try:
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        print(f"\nModelo salvo em: {model_path}")
        print(f"Scaler salvo em: {scaler_path}")
    except Exception as e:
        print(f"Erro ao salvar modelo/scaler: {e}")


if __name__ == "__main__":
    # Define os caminhos relativos para o dataset e para salvar os modelos
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    data_file_path = os.path.join(current_script_dir, '..', 'dataset', 'creditcard.csv')
    model_save_dir = os.path.join(current_script_dir, '..', 'modeloperation')

    train_and_save_model(data_file_path, model_save_dir)

