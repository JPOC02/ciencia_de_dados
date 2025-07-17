import streamlit as st
import pandas as pd
import kagglehub
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Importe o SMOTE (instale com: pip install imbalanced-learn)
from imblearn.over_sampling import SMOTE

# --- Funções de Cache ---

@st.cache_data
def load_data():
    """Baixa e carrega o dataset de diabetes."""
    try:
        path = kagglehub.dataset_download("whenamancodes/predict-diabities")
        csv_path = os.path.join(path, 'diabetes.csv')
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return None

# Função de treinamento modificada para lidar com o desbalanceamento
@st.cache_resource(show_spinner="Treinando o modelo...")
def train_model(df, balancing_strategy='Nenhum'):
    """Prepara os dados, aplica uma estratégia de balanceamento e treina o modelo."""
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Usar stratify
    
    # Aplica a estratégia de balanceamento selecionada APENAS nos dados de TREINO
    if balancing_strategy == 'SMOTE (Oversampling)':
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif balancing_strategy == 'Ponderação de Classes (class_weight)':
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    else: # 'Nenhum'
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Treina o modelo
    model.fit(X_train, y_train)
    
    # Avaliação do modelo nos dados de TESTE (que não foram reamostrados)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Não Diabético', 'Diabético'])
    cm = confusion_matrix(y_test, y_pred)
    
    return model, acc, report, cm


def user_input_features():
    pregnancies = st.sidebar.slider('Gestações', 0, 17, 3)
    glucose = st.sidebar.slider('Glicose', 40, 200, 117)
    blood_pressure = st.sidebar.slider('Pressão Sanguínea', 20, 122, 72)
    skin_thickness = st.sidebar.slider('Espessura da Pele', 7, 99, 29)
    insulin = st.sidebar.slider('Insulina', 14, 846, 125)
    bmi = st.sidebar.slider('IMC', 18.0, 67.0, 32.3, 0.1)
    dpf = st.sidebar.slider('Histórico Familiar', 0.078, 2.42, 0.3725, 0.001)
    age = st.sidebar.slider('Idade', 21, 81, 29)
    data = {'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': blood_pressure, 'SkinThickness': skin_thickness, 'Insulin': insulin, 'BMI': bmi, 'DiabetesPedigreeFunction': dpf, 'Age': age}
    features = pd.DataFrame(data, index=[0])
    return features

# --- Configuração da Página ---
st.set_page_config(page_title="Classificação Interativa", layout="wide")

# --- Carregamento e Limpeza dos Dados (sem alterações) ---
df_original = load_data()
if df_original is None:
    st.stop()

df = df_original.copy()
st.markdown("### Combatendo o Desbalanceamento de Classes")

with st.expander("Análise Exploratória e Tratamento dos Dados"):
    st.header("Análise Exploratória do Dataset")
    st.subheader("Distribuição das Classes (Outcome)")
    class_dist = df['Outcome'].value_counts()
    st.bar_chart(class_dist)
    st.write(f"Amostras 'Sem Diabetes' (0): **{class_dist[0]}**")
    st.write(f"Amostras 'Com Diabetes' (1): **{class_dist[1]}**")
    st.warning("O dataset é desbalanceado, o que pode levar a um modelo com predições enviesadas.")

    cols_to_replace = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in cols_to_replace:
        df[col] = df[col].replace(0, np.nan)
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

# --- Barra Lateral (Sidebar) ---
st.sidebar.header("Insira os Dados do Paciente")


input_df = user_input_features()

# --- Seção de Machine Learning ---
st.markdown("---")
st.header("🔬 Predição e Avaliação do Modelo")

# Seletor de estratégia de balanceamento na sidebar
balancing_strategy = st.sidebar.selectbox(
    'Escolha a Estratégia de Balanceamento:',
    ('Nenhum (Desbalanceado)', 'Ponderação de Classes (class_weight)', 'SMOTE (Oversampling)')
)

# Treina o modelo com a estratégia escolhida
model, acc, report, cm = train_model(df, balancing_strategy)

# Colunas para organizar a exibição
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("Performance do Modelo")
    st.write(f"Estratégia: **{balancing_strategy}**")
    st.write(f"Acurácia no Teste: **{acc:.2%}**")
    st.text_area("Relatório de Classificação", report, height=250)

with col2:
    st.subheader("Matriz de Confusão")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Não Diabético', 'Diabético'], yticklabels=['Não Diabético', 'Diabético'])
    ax.set_xlabel('Predito')
    ax.set_ylabel('Verdadeiro')
    st.pyplot(fig)

st.info("""
**Como ler os resultados:**
- **Relatório:** Foco no **Recall** da classe 'Diabético'. Um valor mais alto significa que o modelo está melhor em identificar quem realmente tem a doença. O **F1-Score** é um bom resumo geral.
- **Matriz de Confusão:** Observe o canto inferior esquerdo (Falsos Negativos). Nosso objetivo é minimizar este número.
""")

# Lógica de predição interativa
if st.sidebar.button("Fazer Predição"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    st.subheader("Resultado da Predição Interativa")
    if prediction[0] == 1:
        st.error("Resultado: **Positivo para Diabetes**", icon="🚨")
    else:
        st.success("Resultado: **Negativo para Diabetes**", icon="✅")
    st.metric(label="Confiança na Predição (Positivo)", value=f"{prediction_proba[0][1]:.2%}")