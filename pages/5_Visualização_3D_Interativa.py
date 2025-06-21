# pages/5_Visualização_3D_Interativa.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
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
st.set_page_config(page_title="Visualização 3D", page_icon="🧊", layout="wide")

# Carrega os dados
df = load_data()
if df is None:
    st.stop()

# --- Conteúdo da Página ---
st.header("Visualização 3D: IMC, Insulina e Idade")
st.write("Este gráfico 3D interativo permite explorar a relação entre IMC, Insulina e Idade, colorindo os pontos pelo resultado de diabetes.")

# Preparação dos dados para o gráfico 3D
df_3d = df[['BMI', 'Insulin', 'Age', 'Outcome']].copy()

# ---- MUDANÇA 1: Tratar 'Outcome' como uma categoria explícita ----
# Isso garante que o Plotly a interprete corretamente para a legenda e as cores.
df_3d['Outcome'] = df_3d['Outcome'].astype('category')
    
# Aplicar transformações para melhor visualização
df_3d['Insulin'] = df_3d['Insulin'] * 5 # Amplifica a insulina para melhor escala visual
    
# Normalização dos dados para o plot
scaler = MinMaxScaler()
df_3d[['BMI', 'Insulin', 'Age']] = scaler.fit_transform(df_3d[['BMI', 'Insulin', 'Age']])
    
# Criação do gráfico com Plotly Express
fig_3d = px.scatter_3d(
    df_3d, 
    x='BMI', y='Insulin', z='Age',
    color='Outcome',
    title='Relação entre IMC, Insulina (x5, Normalizada) e Idade por Resultado',
    labels={'Outcome': 'Resultado'}, # Mantém o rótulo da legenda claro

    # ---- MUDANÇA 2: Usar 'color_discrete_map' para cores categóricas ----
    # Em vez de uma escala contínua, mapeamos cada valor para uma cor específica.
    color_discrete_map={
        0: 'blue',  # Cor para a categoria 0 (Não Diabético)
        1: 'orange' # Cor para a categoria 1 (Diabético)
    }
)
    
# Melhorias no layout
fig_3d.update_layout(
    scene = dict(
        xaxis_title='IMC (Normalizado)',
        yaxis_title='Insulina (x5, Normalizada)',
        zaxis_title='Idade (Normalizada)'),
    margin=dict(l=0, r=0, b=0, t=40)
)
    
st.plotly_chart(fig_3d, use_container_width=True)
st.info("Use o mouse para rotacionar, dar zoom e explorar os dados neste gráfico interativo.")