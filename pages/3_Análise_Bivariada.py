# pages/3_Análise_Bivariada.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
st.set_page_config(page_title="Análise Bivariada", page_icon="🤝", layout="wide")
sns.set_style('whitegrid')

# Carrega os dados
df = load_data()
if df is None:
    st.stop()

# Tratando os dados faltantes

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


# --- Conteúdo da Página ---
st.header("Análise Bivariada")
st.write("Analisando a relação entre duas variáveis.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Pressão Sanguínea por Resultado")
    fig_bp, ax_bp = plt.subplots()
    sns.boxplot(x='Outcome', y='BloodPressure', data=df, ax=ax_bp)
    ax_bp.set_title('Distribuição da Pressão Sanguínea por Resultado')
    ax_bp.set_xlabel('Resultado')
    ax_bp.set_ylabel('Pressão Sanguínea')
    st.pyplot(fig_bp)

with col2:
    st.subheader("IMC vs. Glicose por Resultado")
    fig_scatter, ax_scatter = plt.subplots()
    sns.scatterplot(x='BMI', y='Glucose', hue='Outcome', data=df, alpha=0.6, ax=ax_scatter)
    ax_scatter.set_title('IMC vs. Glicose por Resultado')
    ax_scatter.set_xlabel('IMC')
    ax_scatter.set_ylabel('Glicose')
    st.pyplot(fig_scatter)
    
st.info("A análise bivariada ajuda a identificar relações e padrões entre pares de variáveis.")