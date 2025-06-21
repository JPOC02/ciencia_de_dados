# pages/2_Análise_Univariada.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
import os

# Função de carregamento de dados (com cache)
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("whenamancodes/predict-diabities")
    csv_path = os.path.join(path, 'diabetes.csv')
    df = pd.read_csv(csv_path)
    return df

# --- Configuração da Página ---
st.set_page_config(page_title="Análise Univariada", page_icon="📈", layout="wide")
sns.set_style('whitegrid')

# Carrega os dados
df = load_data()
if df is None:
    st.stop()

# --- Conteúdo da Página ---
st.header("Análise Univariada")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribuição da Idade")
    fig_age, ax_age = plt.subplots()
    sns.histplot(df['Age'], bins=20, kde=True, ax=ax_age)
    ax_age.set_title('Distribuição da Idade')
    ax_age.set_xlabel('Idade')
    ax_age.set_ylabel('Frequência')
    st.pyplot(fig_age)

with col2:
    st.subheader("Distribuição do Resultado (Diabetes)")
    fig_outcome, ax_outcome = plt.subplots()
    sns.countplot(x='Outcome', data=df, ax=ax_outcome)
    ax_outcome.set_title('Distribuição do Resultado de Diabetes')
    ax_outcome.set_xlabel('Resultado (0: Não Diabético, 1: Diabético)')
    ax_outcome.set_ylabel('Contagem')
    st.pyplot(fig_outcome)
    
st.info("A análise univariada foca em uma variável por vez para entender sua distribuição.")