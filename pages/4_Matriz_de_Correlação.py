# pages/4_Matriz_de_Correlação.py
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
st.set_page_config(page_title="Matriz de Correlação", page_icon="🔗", layout="wide")
sns.set_style('whitegrid')

# Carrega os dados
df = load_data()
if df is None:
    st.stop()

# --- Conteúdo da Página ---
st.header("Matriz de Correlação das Variáveis")
st.write("A matriz de correlação mostra a força e a direção da relação linear entre as variáveis.")

st.subheader("Correlação de Todas as Variáveis")
fig_corr, ax_corr = plt.subplots(figsize=(12, 9))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
st.pyplot(fig_corr)
st.markdown("""
**Interpretação:**
- Valores próximos de **+1** indicam uma forte correlação positiva.
- Valores próximos de **-1** indicam uma forte correlação negativa.
- Valores próximos de **0** indicam pouca ou nenhuma correlação linear.
""")

st.subheader("Correlação Excluindo Registros com Insulina = 0")
st.write("Muitos valores de 'Insulin' são 0, o que pode não ser fisiologicamente possível e distorcer a correlação. Vamos analisar a matriz sem eles.")
    
df_filtered = df[df['Insulin'] != 0]
fig_corr_filtered, ax_corr_filtered = plt.subplots(figsize=(12, 9))
correlation_matrix_filtered = df_filtered.corr()
sns.heatmap(correlation_matrix_filtered, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr_filtered)
st.pyplot(fig_corr_filtered)
st.info("Observe como a correlação entre Insulina e Glicose se torna muito mais forte (de 0.13 para 0.67) após filtrar os zeros.")