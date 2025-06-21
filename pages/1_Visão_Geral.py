# pages/1_Visão_Geral.py
import streamlit as st
import pandas as pd
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
st.set_page_config(page_title="Visão Geral", page_icon="📊", layout="wide")

# Carrega os dados
df = load_data()
if df is None:
    st.stop()

# --- Conteúdo da Página ---
st.header("Visão Geral do Dataset")
    
st.subheader("Primeiras Linhas do Dataset")
st.dataframe(df.head())
    
st.subheader("Formato do Dataset")
st.write(f"O dataset possui **{df.shape[0]} linhas** e **{df.shape[1]} colunas**.")
    
st.subheader("Tipos de Dados por Coluna")
dtypes_df = df.dtypes.to_frame('Tipo de Dado').astype(str)
st.dataframe(dtypes_df)

st.subheader("Resumo Estatístico")
st.dataframe(df.describe())
    
st.subheader("Contagem de Valores Faltantes")
missing_values = df.isnull().sum().to_frame('Valores Faltantes')
st.dataframe(missing_values)
if missing_values.sum().iloc[0] == 0:
    st.success("Ótima notícia! Não há valores faltantes no dataset.")