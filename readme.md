---
# Detecção de Fraudes em Cartão de Crédito

Este projeto implementa uma solução inicial para a detecção de transações fraudulentas em cartão de crédito, utilizando Machine Learning. A solução abrange desde o treinamento do modelo até a sua disponibilização via uma API de back-end e uma interface de usuário simples.

---

## 1. Introdução ao Projeto

A detecção de fraudes em transações financeiras é um desafio crítico para bancos e instituições financeiras. O volume massivo de transações diárias e a natureza rara das fraudes (uma agulha no palheiro!) exigem soluções inteligentes e eficientes. Este projeto visa construir um "detetive digital" capaz de identificar padrões suspeitos em transações de cartão de crédito e classificá-las como legítimas ou fraudulentas, minimizando prejuízos e protegendo usuários.

---

## 2. Objetivo

O principal objetivo deste projeto é:

* Desenvolver um **modelo de Machine Learning robusto** para classificar transações de cartão de crédito como fraudulentas ou não fraudulentas.
* Construir uma **API** que permita que outras aplicações (como um front-end) enviem dados de transações e recebam instantaneamente a classificação do modelo.
* Fornecer uma **interface de usuário simples** para demonstrar a funcionalidade da solução, permitindo a entrada de dados de transação e a visualização do veredito do modelo.

---

## 3. Dataset Popular

* **Credit Card Fraud Detection (Kaggle)**: Este é um dos datasets mais utilizados para estudos de detecção de fraudes. Ele contém transações de cartão de crédito europeias em setembro de 2013.
* **Características**:
    * Contém **284.807 transações**, das quais apenas **492 são fraudulentas (0.172% da base)**. Isso ilustra o problema do desbalanceamento.
    * A maioria das features (V1, V2, ..., V28) são resultado de uma **Transformação de Componentes Principais (PCA)** para preservar a privacidade dos dados.
    * As features `Time` (segundos desde a primeira transação) e `Amount` (valor da transação) não foram transformadas.
    * A coluna `Class` é a variável alvo: **0 para transações legítimas e 1 para fraudulentas**.
* **Onde encontrar**: Você pode baixá-lo diretamente do Kaggle: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 4. Ferramentas, Tecnologias e Bibliotecas Utilizadas

A solução é construída com uma combinação de ferramentas e bibliotecas populares nas áreas de Machine Learning e desenvolvimento web.

### 4.1. Machine Learning & Análise de Dados (Python)

* **Python**: Linguagem de programação principal para o desenvolvimento do modelo e do back-end.
* **Jupyter Notebook**: Ambiente interativo de desenvolvimento (IDE) utilizado para a exploração de dados, prototipagem, treinamento e avaliação do modelo. Ideal para a fase de pesquisa e desenvolvimento.
* **Bibliotecas Python**:
    * **Pandas**: Manipulação e análise de dados (leitura de CSV, organização de tabelas).
    * **Scikit-learn**: Ferramentas essenciais para Machine Learning, incluindo pré-processamento de dados (`StandardScaler`), divisão de conjuntos de dados (`train_test_split`), algoritmos de classificação (`RandomForestClassifier`) e métricas de avaliação (`classification_report`, `roc_auc_score`).
    * **Imbalanced-learn**: Extensão do Scikit-learn, crucial para lidar com datasets desbalanceados (poucas fraudes vs. muitas transações normais) através de técnicas como SMOTE.
    * **Joblib**: Para persistir (salvar) o modelo treinado e o scaler, permitindo que sejam carregados e utilizados pela API.

### 4.2. Back-end (API - Python)

* **Flask**: Micro-framework web leve e flexível em Python, utilizado para construir a API RESTful que serve o modelo de Machine Learning.

### 4.3. Front-end (Interface Web)

* **HTML**: Estrutura da página web.
* **CSS**: Estilização da interface para uma melhor experiência do usuário.
* **JavaScript**: Lógica interativa do lado do cliente, responsável por capturar os dados do formulário, enviar requisições para a API e exibir os resultados.

---

## 5. Organização das Pastas do Projeto

O projeto segue a seguinte estrutura de diretórios para facilitar a organização e a manutenção:

.
├── notebook/
│   └── fraud_detection.py  # Script que simula Notebook para exploração e treinamento do modelo.
├── backend/
│   └── app.py                 # Código da API Flask.
├── frontend/
│   └── index.html             # Página HTML com o formulário e exibição de resultados.
├── modeloperation/
│   ├── fraud_detection_model.pkl  # Modelo de ML treinado e salvo.
│   └── scaler.pkl                 # Objeto StandardScaler salvo (para pré-processar novas entradas).
│   └── processor.py               # Pipeline de execução do modelo treinado.
├── dataset/
│   └── creditcard.csv         # Dataset original de transações de cartão de crédito (precisa baixar).
├── test/
│   └── test_app.py            # Script para testes unitários.
├── README.md                  # Este arquivo.
└── requirements.txt           # Dependências Python do projeto.

---

## 6. Forma de Execução do Projeto

Para colocar a solução em funcionamento, siga os passos abaixo:

### 6.1. Pré-requisitos

* **Python 3.8+** instalado.
* **Pip** (gerenciador de pacotes do Python) instalado.
* **PyCharm** ou outro IDE com suporte a Python e Jupyter.

### 6.2. Configuração do Ambiente

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/giseleluti/fraud_detection
    cd fraud_detection
    ```
2.  **Crie e ative um ambiente virtual:**
    ```bash
    python -m venv venv_fraude
    # No Windows:
    .\venv_fraude\Scripts\activate
    # No macOS/Linux:
    source venv_fraude/bin/activate
    ```
3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

### 6.3. Preparação do Dataset

1.  **Baixe o dataset `creditcard.csv` do Kaggle**: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2.  Descompacte o arquivo em formato CSV.
3.  Coloque o arquivo `creditcard.csv` dentro da pasta `dataset/`.

### 6.4. Treinamento e Salvamento do Modelo

1.  **Inicie o script que simula o notebook ** (no PyCharm), ou executar a pipeline **processor.py**:
O processor.py está na pasta `/modelOperation/` que já contem o modelo treinado inclusive
2.  Navegue até a pasta `notebook/` e abra o arquivo `fraud_detection.py`.
3.  **Execute todas as células do notebook sequencialmente**. Este notebook irá:
    * Carregar e pré-processar os dados.
    * Realizar a engenharia de features e lidar com o desbalanceamento.
    * Treinar o modelo de detecção de fraudes.
    * Avaliar o desempenho do modelo.
    * Salvar o modelo treinado (`fraud_detection_model.pkl`) e o scaler (`scaler.pkl`) na pasta `modeloperation/`.

### 6.5. Execução do Back-end (API)

1.  Certifique-se de que seu ambiente virtual esteja ativado.
2.  Navegue até a pasta `backend/`.
3.  Execute o script da API:
    ```bash
    python app.py
    ```
    A API será iniciada, geralmente em `http://127.0.0.1:5000` (ou `http://localhost:5000`).

### 6.6. Execução do Front-end

1.  Abra o navegador web e digite diretamente `http://127.0.0.1:5000` ou `http://localhost:5000`. Você será direcionado para o frontend.
2.  Com a API de back-end em execução, você poderá inserir dados de transação no formulário e submetê-los para obter a classificação do modelo.
3.  Devido à singularidade e à natureza real dos dados, utilize os exemplos indicados no front-end: os 3 primeiros são transações não-fraude e os 3 últimos são fraude, todos extraídos do dataset original.

---

## 7. Testes

* **Testes Unitários**: Para verificar se funções individuais de pré-processamento ou partes do modelo funcionam como esperado.

---

## 8. Relatório de Treinamento (Exemplo)

Ao executar o `processor.py` (pipeline) com o algoritmo pré-selecionado, você verá um relatório de treino similar a este:

Iniciando treinamento do modelo RandomForest...
Treinamento concluído.
--- Relatório de Classificação no conjunto de Teste ---
precision    recall  f1-score   support
Não Fraude       1.00      1.00      1.00     56864
Fraude       0.87      0.83      0.85        98
accuracy                           1.00     56962
macro avg       0.94      0.91      0.92     56962
weighted avg       1.00      1.00      1.00     56962

---

## 9. Na Prática

Essa API estaria integrada diretamente aos sistemas de transação de uma instituição financeira e não a um front-end de demonstração.

Quando uma transação acontece:

1.  Os dados brutos (localização, tipo de compra, dispositivo, IP, etc.) são coletados automaticamente pelos sistemas da instituição.
2.  Esses dados brutos são transformados internamente pela instituição (usando o mesmo processo de PCA ou outro) para gerar as V features.
3.  As features V1 a V28 (junto com Time e Amount) são então enviadas automaticamente para a API de detecção de fraudes.
4.  A API retorna o veredito, e essa informação é usada para decidir se a transação é aprovada, negada ou enviada para revisão manual.