# pages/4_Matriz_de_Correlação.py
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
st.set_page_config(page_title="Matriz de Correlação", page_icon="🔗", layout="wide")
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
st.header("Matriz de Correlação das Variáveis")
st.write("A matriz de correlação mostra a força e a direção da relação entre as variáveis.")

st.subheader("Correlação de Todas as Variáveis")
fig_corr, ax_corr = plt.subplots(figsize=(12, 9))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
st.pyplot(fig_corr)
st.markdown("""
**Interpretação:**
- Valores próximos de **+1** indicam uma forte correlação positiva.
- Valores próximos de **-1** indicam uma forte correlação negativa.
- Valores próximos de **0** indicam pouca ou nenhuma correlação.
""")