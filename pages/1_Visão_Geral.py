# pages/1_Visão_Geral.py
import streamlit as st
import pandas as pd
import kagglehub
import os
import numpy as np

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

# --- Displaying the Cleaned Data ---
st.subheader("Primeiras Linhas do Dataset")
st.dataframe(df.head())
    
st.subheader("Formato do Dataset")
st.write(f"O dataset possui **{df.shape[0]} linhas** e **{df.shape[1]} colunas**.")
    
st.subheader("Tipos de Dados por Coluna")
dtypes_df = df.dtypes.to_frame('Tipo de Dado').astype(str)
st.dataframe(dtypes_df)



st.subheader("Resumo Estatístico")
st.info("Dados antes do tratamento")
st.dataframe(df.describe())

# --- Data Cleaning and Preprocessing Section ---
st.subheader("Tratamento de Valores Nulos (Zeros)")
st.write("""
Valores iguais a zero em certas colunas (como Glicose, Pressão Sanguínea, IMC, etc.) 
são clinicamente improváveis e podem ser considerados dados ausentes ou inválidos. 
Para tratar isso, substituímos os zeros pela **mediana** da respectiva coluna, que é uma 
medida de tendência central robusta a outliers.
""")

# Lista de colunas onde os zeros seram substituidos pela mediana
cols_to_replace = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
# Dictionario para guardar as alterações feitas
replacements_log = {}

# Loop que fará a alteração em cada coluna
for col in cols_to_replace:
    # Contador do número de zeros na coluna
    zero_count = (df[col] == 0).sum()
    
    # Só avança se houverem zeros para substituir
    if zero_count > 0:
        # Substitui inicialmente os 0s por NaN para realizar o cálculo correto da mediana
        # A função de mediana ignora valores iguais a NaN
        df[col] = df[col].replace(0, np.nan)
        
        # Calculando a mediana
        median_val = df[col].median()
        
        # Substituindo os valores faltantes pela mediana
        df[col] = df[col].fillna(median_val)
        
        # Salvando o número de alterações e a mediana calculada
        replacements_log[col] = {
            'Zeros Substituídos': zero_count,
            'Valor da Mediana Usado': median_val
        }

# Apresentando um resumo das alterações
if replacements_log:
    st.info("Resumo das substituições realizadas:")
    summary_df = pd.DataFrame.from_dict(replacements_log, orient='index')
    st.dataframe(summary_df)
else:
    st.info("Não foram encontrados valores zero para substituir nas colunas especificadas.")

st.info("Dados após o tratamento")
st.dataframe(df.describe())
    