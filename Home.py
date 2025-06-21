# Home.py
import streamlit as st

# --- Configuração da Página ---
# É bom definir a configuração em cada página para consistência
st.set_page_config(
    page_title="Análise de Diabetes | Home",
    page_icon="🩺",
    layout="wide"
)

# --- Conteúdo da Página Principal ---
st.title("🩺 Análise Exploratória de Dados de Diabetes")

st.markdown("""
Esta aplicação interativa foi criada para analisar o dataset de previsão de diabetes.
Explore as diferentes seções analíticas usando o menu de navegação na barra lateral à esquerda.

### O que você encontrará:
- **Visão Geral do Dataset:** Informações básicas, resumo estatístico e tipos de dados.
- **Análise Univariada:** Gráficos que exploram cada variável individualmente.
- **Análise Bivariada:** Relações entre pares de variáveis importantes.
- **Matriz de Correlação:** Uma visão geral das correlações lineares no dataset.
- **Visualização 3D:** Um gráfico interativo para explorar a relação entre IMC, Insulina e Idade.

**Para começar, selecione uma página no menu de navegação.**
""")

st.info("Todos os dados são carregados dinamicamente a partir do Kaggle e mantidos em cache para uma navegação rápida entre as páginas.")