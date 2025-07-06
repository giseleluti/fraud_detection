##%
"""
# Notebook de Detecção de Fraudes em Cartão de Crédito - por Gisele Priscila

Este notebook documenta o processo completo de construção e avaliação de um modelo de Machine Learning para detecção de transações fraudulentas em cartão de crédito.
Aqui constam  etapas essenciais de um pipeline de ML, desde a carga e pré-processamento dos dados até a modelagem, otimização e avaliação de diferentes algoritmos de classificação.
"""

##%
"""
## 1. Contexto do Problema

A detecção de fraudes em transações financeiras é um desafio crítico para bancos e instituições financeiras. Com o volume crescente de transações digitais, a capacidade de identificar rapidamente atividades suspeitas é fundamental para mitigar perdas financeiras e manter a confiança dos clientes.

Este projeto foca em um problema de **classificação binária**, onde o objetivo é classificar cada transação como:
* **0: Transação Legítima** (não fraudulenta)
* **1: Transação Fraudulenta**

Um dos maiores desafios neste tipo de problema é o **desbalanceamento de classes**: o número de transações fraudulentas é extremamente pequeno em comparação com o número de transações legítimas. Isso exige abordagens cuidadosas tanto no pré-processamento quanto na avaliação do modelo.
"""

##%
"""
## 2. Carga e Separação dos Dados

Nesta etapa, carregamos o dataset de transações de cartão de crédito e o dividimos em conjuntos de treino e teste. A divisão é crucial para garantir que avaliamos o modelo em dados que ele **nunca viu** durante o treinamento, evitando que ele apenas "decore" os exemplos de treino.

O dataset utilizado foi obtido no Kaggle, contendo transações europeias de setembro de 2013. As features V1 a V28 são resultado de uma Transformação de Componentes Principais (PCA) para proteger a privacidade dos dados originais. As features Time e Amount não foram transformadas.
"""

##%
"""
### 2.1. Importação de Bibliotecas Essenciais
"""
# Imports necessários
import pandas as pd # Para manipulação e análise de dados (DataFrames)
from sklearn.model_selection import train_test_split # Para dividir o dataset em treino e teste

##%
"""
### 2.2. Carregamento do Dataset (Local)

**ATENÇÃO:** O dataset `creditcard.csv` deve estar na pasta `dataset/`
"""
import os

# Caminho para o dataset localmente
dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset', 'creditcard.csv')

# Carrega o dataset para um DataFrame do pandas
try:
    df = pd.read_csv(dataset_path, delimiter=",")
    print("Dataset carregado com sucesso!")
    print(f"Formato do dataset: {df.shape}")
    print("\nPrimeiras 5 linhas do dataset:")
    print(df.head())
except FileNotFoundError:
    print(f"Erro: O arquivo '{dataset_path}' não foi encontrado.")
    print("Por favor, certifique-se de que o dataset 'creditcard.csv' está na pasta 'dataset/' na raiz do projeto.")
except Exception as e:
    print(f"Erro ao carregar o dataset: {e}")
    print("Verifique se o caminho e o formato do arquivo estão corretos.")

##%
# Informações gerais do dataset
print("\nInformações gerais do dataset:")
df.info()

##%
"""
### 2.3. Separação de Features (X) e Target (y)
"""
# 'Class' é a variável alvo: 0 para não fraude, 1 para fraude
X = df.drop('Class', axis=1) # X contém todas as colunas, exceto 'Class'
y = df['Class']             # y contém apenas a coluna 'Class'

print(f"Formato de X (features): {X.shape}")
print(f"Formato de y (target): {y.shape}")
print("\nDistribuição da classe 'Class' (original):")
print(y.value_counts(normalize=True) * 100) # Mostra a porcentagem de cada classe

"""
A distribuição da classe `Class` nos mostra o **desbalanceamento** dos dados: a porcentagem de fraudes (1) é muito baixa.
"""

##%
"""
### 2.4. Divisão em Conjuntos de Treino e Teste (Holdout)

A função `train_test_split` para dividir os dados em 80% para treinamento e 20% para teste. O parâmetro `stratify=y` é crucial aqui, pois garante que a proporção de transações fraudulentas seja mantida nos dois conjuntos, o que é vital para lidar com o desbalanceamento.
"""
# Dividir o dataset em treino (80%) e teste (20%)
# random_state garante que a divisão seja a mesma cada vez que o código for executado
# stratify=y garante que a proporção de classes (fraude/não fraude) seja a mesma nos conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nFormato de X_train: {X_train.shape}")
print(f"Formato de X_test: {X_test.shape}")
print(f"Formato de y_train: {y_train.shape}")
print(f"Formato de y_test: {y_test.shape}")

print("\nDistribuição da classe 'Class' no conjunto de treino:")
print(y_train.value_counts(normalize=True) * 100)
print("\nDistribuição da classe 'Class' no conjunto de teste:")
print(y_test.value_counts(normalize=True) * 100)

##%
"""
## 3. Transformação de Dados: Padronização

O `scaler` é **ajustado (fit)** apenas nos dados de treinamento e depois **transforme** tanto os dados de treino quanto os de teste usando o mesmo `scaler`.
"""

##%
"""
### 3.1. Importação do StandardScaler
"""
from sklearn.preprocessing import StandardScaler # Para padronizar os dados

##%
"""
### 3.2. Aplicação do StandardScaler
"""
# Cria uma instância do StandardScaler
scaler = StandardScaler()

# Ajusta o scaler aos dados de treino e os transforma
X_train_scaled = scaler.fit_transform(X_train)

# Apenas transforma os dados de teste usando o scaler ajustado nos dados de treino
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

print("\nDados de treino padronizados (primeiras 5 linhas):")
print(X_train_scaled_df.head())

##%
"""
## 4. Lidando com o Desbalanceamento de Classes (SMOTE)

O **SMOTE (Synthetic Minority Over-sampling Technique)** é uma técnica de superamostragem que cria exemplos sintéticos da classe minoritária, ajudando o modelo a aprender melhor os padrões de fraude.
"""

##%
"""
### 4.1. Importação do SMOTE
"""
from imblearn.over_sampling import SMOTE # Para lidar com o desbalanceamento de classes

##%
"""
### 4.2. Aplicação do SMOTE no Conjunto de Treino
"""
# Cria uma instância do SMOTE
smote = SMOTE(random_state=42)

# Aplica o SMOTE no conjunto de treino padronizado
X_train_res, y_train_res = smote.fit_resample(X_train_scaled_df, y_train)

print(f"Formato de X_train após SMOTE: {X_train_res.shape}")
print(f"Formato de y_train após SMOTE: {y_train_res.shape}")
print("\nDistribuição da classe 'Class' no conjunto de treino após SMOTE:")
print(y_train_res.value_counts(normalize=True) * 100)

"""
A classe de fraude (1) tem uma representação muito mais equitativa no conjunto de treinamento.
"""

##%
"""
## 5. Modelagem: Treinamento e Comparação de Algoritmos

Nesta seção, treinaremos diferentes modelos de classificação utilizando os seguintes algoritmos: 
**KNN**, **Árvore de Classificação**, **Naive Bayes** e **SVM**. 
Para cada modelo, haverá  a otimização de hiperparâmetros (usando `GridSearchCV` para uma busca exaustiva) e a avaliação do desempenho usando `cross-validation`.

**Observação:** O `StandardScaler` foi aplicado **antes** do SMOTE, e o SMOTE nos dados de treino.

"""

##%
"""
### 5.1. Importação de Algoritmos e Ferramentas de Avaliação
"""
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.tree import DecisionTreeClassifier    # Árvore de Classificação
from sklearn.naive_bayes import GaussianNB         # Naive Bayes
from sklearn.svm import SVC                        # SVM (Support Vector Machine)

from sklearn.model_selection import GridSearchCV, StratifiedKFold # Para otimização de hiperparâmetros e validação cruzada
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score # Métricas de avaliação
import matplotlib.pyplot as plt # Para visualização
import seaborn as sns # Para visualização avançada

##%
"""
### 5.2. Métricas de Avaliação Personalizadas

Uma função auxiliar para imprimir as métricas mais importantes para a detecção de fraude.
"""
def evaluate_model(model, X_test, y_test, model_name="Modelo"):
    """
    Avalia o desempenho do modelo no conjunto de teste e imprime métricas relevantes.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    print(f"\n--- Avaliação do {model_name} ---")
    print(classification_report(y_test, y_pred, target_names=['Não Fraude', 'Fraude']))

    # Métricas específicas para a classe 'Fraude' (classe 1)
    precision_fraud = precision_score(y_test, y_pred, pos_label=1)
    recall_fraud = recall_score(y_test, y_pred, pos_label=1)
    f1_fraud = f1_score(y_test, y_pred, pos_label=1)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 'N/A'

    print(f"Precisão (Fraude): {precision_fraud:.4f}")
    print(f"Recall (Fraude): {recall_fraud:.4f}")
    print(f"F1-Score (Fraude): {f1_fraud:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")

    return {
        'Model': model_name,
        'Precision_Fraud': precision_fraud,
        'Recall_Fraud': recall_fraud,
        'F1_Score_Fraud': f1_fraud,
        'AUC_ROC': roc_auc
    }

results_comparison = [] # Lista para armazenar os resultados de todos os modelos

##%
"""
### 5.3. Treinamento e Otimização para Cada Algoritmo

"""
# Configuração da validação cruzada
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- 5.3.1. K-Nearest Neighbors (KNN) ---
print("\n##### Treinando e Otimizando KNN #####")
knn_model = KNeighborsClassifier()
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9], # Número de vizinhos
    'weights': ['uniform', 'distance'] # Peso dos vizinhos
}

grid_search_knn = GridSearchCV(knn_model, knn_param_grid, cv=cv_strategy, scoring='f1', n_jobs=-1, verbose=1)
grid_search_knn.fit(X_train_res, y_train_res) # Treina com dados balanceados

best_knn_model = grid_search_knn.best_estimator_
print(f"Melhores parâmetros para KNN: {grid_search_knn.best_params_}")
results_comparison.append(evaluate_model(best_knn_model, X_test_scaled_df, y_test, "KNN"))


# --- 5.3.2. Árvore de Classificação (Decision Tree) ---
print("\n##### Treinando e Otimizando Árvore de Classificação #####")
dt_model = DecisionTreeClassifier(random_state=42, class_weight='balanced') # class_weight para dados desbalanceados
dt_param_grid = {
    'max_depth': [None, 5, 10, 15], # Profundidade máxima da árvore
    'min_samples_split': [2, 5, 10], # Número mínimo de amostras para dividir um nó
    'min_samples_leaf': [1, 2, 4] # Número mínimo de amostras em uma folha
}

grid_search_dt = GridSearchCV(dt_model, dt_param_grid, cv=cv_strategy, scoring='f1', n_jobs=-1, verbose=1)
grid_search_dt.fit(X_train_res, y_train_res)
best_dt_model = grid_search_dt.best_estimator_
print(f"Melhores parâmetros para Árvore de Classificação: {grid_search_dt.best_params_}")
results_comparison.append(evaluate_model(best_dt_model, X_test_scaled_df, y_test, "Árvore de Classificação"))


# --- 5.3.3. Naive Bayes (Gaussian Naive Bayes) ---
print("\n##### Treinando Naive Bayes #####")
# Naive Bayes geralmente não tem muitos hiperparâmetros para otimização em GridSearchCV
gnb_model = GaussianNB()
gnb_model.fit(X_train_res, y_train_res)
results_comparison.append(evaluate_model(gnb_model, X_test_scaled_df, y_test, "Naive Bayes"))


# --- 5.3.4. Support Vector Machine (SVM) ---
print("\n##### Treinando e Otimizando SVM (Pode levar tempo!) #####")
# SVMs podem ser lentos em grandes datasets, especialmente com GridSearch
svm_model = SVC(probability=True, random_state=42, class_weight='balanced') # probability=True para predict_proba
svm_param_grid = {
    'C': [0.1, 1, 10], # Parâmetro de regularização
    'gamma': ['scale', 'auto'], # Coeficiente do kernel
    'kernel': ['rbf'] # Tipo de kernel (rbf é comum)
}

grid_search_svm = GridSearchCV(svm_model, svm_param_grid, cv=cv_strategy, scoring='f1', n_jobs=-1, verbose=1)
grid_search_svm.fit(X_train_res, y_train_res)
best_svm_model = grid_search_svm.best_estimator_
print(f"Melhores parâmetros para SVM: {grid_search_svm.best_params_}")
results_comparison.append(evaluate_model(best_svm_model, X_test_scaled_df, y_test, "SVM"))


# --- 5.3.5. Random Forest ---
print("\n##### Treinando e Otimizando Random Forest #####")
from sklearn.ensemble import RandomForestClassifier # Importar Random Forest
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced') # class_weight para dados desbalanceados
rf_param_grid = {
    'n_estimators': [50, 100, 200], # Número de árvores
    'max_depth': [None, 5, 10], # Profundidade máxima da árvore
    'min_samples_split': [2, 5]
}

grid_search_rf = GridSearchCV(rf_model, rf_param_grid, cv=cv_strategy, scoring='f1', n_jobs=-1, verbose=1)
grid_search_rf.fit(X_train_res, y_train_res)
best_rf_model = grid_search_rf.best_estimator_
print(f"Melhores parâmetros para Random Forest: {grid_search_rf.best_params_}")
results_comparison.append(evaluate_model(best_rf_model, X_test_scaled_df, y_test, "Random Forest"))

##%
"""
### 5.4. Comparação de Resultados Finais

"""
results_df = pd.DataFrame(results_comparison)
print("\n--- Comparação Final dos Modelos ---")
print(results_df.sort_values(by='F1_Score_Fraud', ascending=False))

# Visualizar os resultados
# As figuras serão exibidas na Python Interactive Window do VS Code
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='F1_Score_Fraud', data=results_df.sort_values(by='F1_Score_Fraud', ascending=False))
plt.title('Comparação de F1-Score (Fraude) entre Modelos')
plt.ylabel('F1-Score para Fraude')
plt.ylim(0, 1)
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='AUC_ROC', data=results_df.sort_values(by='AUC_ROC', ascending=False))
plt.title('Comparação de AUC-ROC entre Modelos')
plt.ylabel('AUC-ROC')
plt.ylim(0, 1)
plt.show()

best_overall_model = best_rf_model
print(f"\nModelo escolhido para exportação: {best_overall_model.__class__.__name__}")

##%
"""
## 6. Exportação do Modelo Resultante

"""

##%
"""
### 6.1. Importação do Joblib
"""
import joblib # Para salvar e carregar objetos Python

##%
"""
### 6.2. Salvando o Melhor Modelo e o Scaler

"""
# Caminhos para salvar os arquivos
# Este script está na pasta 'notebook/' e os modelos irão para '../modeloperation/'
model_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'modeloperation')
os.makedirs(model_output_dir, exist_ok=True) # Cria a pasta se ela não existir

model_path = os.path.join(model_output_dir, 'fraud_detection_model.pkl')
scaler_path = os.path.join(model_output_dir, 'scaler.pkl')

try:
    joblib.dump(best_overall_model, model_path)
    joblib.dump(scaler, scaler_path) # Salva o scaler que foi fitado em X_train original
    print(f"\nModelo '{best_overall_model.__class__.__name__}' salvo em: {model_path}")
    print(f"Scaler salvo em: {scaler_path}")
except Exception as e:
    print(f"Erro ao salvar modelo/scaler: {e}")
    print("Verifique as permissões de escrita na pasta 'modeloperation/'.")

##%
"""
## 7. Reflexão sobre Segurança e Anonimização de Dados

No contexto de detecção de fraudes, a segurança e a privacidade dos dados são de extrema importância. A utilização de dados anonimizados, como as features `V1` a `V28` transformadas por PCA no dataset, é uma prática fundamental de **anonimização de dados**.

* **Anonimização por PCA:** A Análise de Componentes Principais (PCA) permite que o modelo aprenda padrões nos dados sem ter acesso às informações sensíveis originais (como números de cartão, nomes de clientes, locais exatos de compra, etc.). Isso protege a privacidade do indivíduo, pois as features transformadas não podem ser facilmente revertidas para os dados originais.
"""