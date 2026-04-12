import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# título
st.title('📊 Dashboard de Análise de Crédito')

# carregar dados
dados = pd.read_csv('../data/bankloans.csv')

# sidebar (filtros)
st.sidebar.header('Filtros')

idade_min = st.sidebar.slider('Idade mínima', int(dados['age'].min()), int(dados['age'].max()), int(dados['age'].min()))
idade_max = st.sidebar.slider('Idade máxima', int(dados['age'].min()), int(dados['age'].max()), int(dados['age'].max()))

renda_min, renda_max = st.sidebar.slider(
    'Renda',
    int(dados['income'].min()),
    int(dados['income'].max()),
    (int(dados['income'].min()), int(dados['income'].max()))
)

educacao = st.sidebar.multiselect(
    'Nível de Educação',
    options=sorted(dados['ed'].unique()),
    default=sorted(dados['ed'].unique())
)

dti_min, dti_max = st.sidebar.slider(
    'Debt-to-Income Ratio',
    float(dados['debtinc'].min()),
    float(dados['debtinc'].max()),
    (float(dados['debtinc'].min()), float(dados['debtinc'].max()))
)

cred_min, cred_max = st.sidebar.slider(
    'Credit-to-Debt Ratio',
    float(dados['creddebt'].min()),
    float(dados['creddebt'].max()),
    (float(dados['creddebt'].min()), float(dados['creddebt'].max()))
)

default_opcao = st.sidebar.selectbox(
    'Clientes',
    ['Todos', 'Confiáveis', 'Não confiáveis']
)

# Converter para valor numérico depois
if default_opcao == 'Confiáveis':
    valor = 0
elif default_opcao == 'Não confiáveis':
    valor = 1

dados_filtrados = dados[
    (dados['age'] >= idade_min) & (dados['age'] <= idade_max) &
    (dados['income'] >= renda_min) & (dados['income'] <= renda_max) &
    (dados['debtinc'] >= dti_min) & (dados['debtinc'] <= dti_max) &
    (dados['creddebt'] >= cred_min) & (dados['creddebt'] <= cred_max) &
    (dados['ed'].isin(educacao))
]

if default_opcao != 'Todos':
    dados_filtrados = dados_filtrados[dados_filtrados['default'] == valor]
    
# métricas
st.subheader('Métricas')
col1, col2 = st.columns(2)

with col1:
    st.metric('Total de Registros', dados_filtrados.shape[0])

with col2:
    taxa_default = dados_filtrados['default'].mean()
    st.metric('Taxa de Default', f'{taxa_default:.2%}')

# ----------------------------
# GRÁFICOS PRINCIPAIS
# ----------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader('Distribuição de Idade')
    fig, ax = plt.subplots()
    sns.histplot(dados_filtrados['age'], bins=30, ax=ax, kde=True)
    ax.set_xlabel('Idade')
    ax.set_ylabel('Frequência')
    st.pyplot(fig)

with col2:
    st.subheader('Distribuição de Confiáveis (0) vs Não confiáveis (1)')
    fig, ax = plt.subplots()
    dados_filtrados['default'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
    ax.set_ylabel('')
    st.pyplot(fig)

# ----------------------------
# NOVOS GRÁFICOS
# ----------------------------

st.subheader('Outras Distribuições')

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.histplot(data=dados_filtrados, x='employ', bins=30, kde=True, ax=ax)
    ax.set_title('Experiência de Trabalho')
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.histplot(data=dados_filtrados, x='creddebt', bins=30, kde=True, ax=ax)
    ax.set_title('Credit-to-Debt Ratio')
    st.pyplot(fig)

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.histplot(data=dados_filtrados, x='othdebt', bins=30, kde=True, ax=ax)
    ax.set_title('Outras Dívidas')
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.histplot(data=dados_filtrados, x='debtinc', bins=30, kde=True, ax=ax)
    ax.set_title('Debt-to-Income Ratio')
    st.pyplot(fig)

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.histplot(data=dados_filtrados, x='income', bins=30, kde=True, ax=ax)
    ax.set_title('Renda')
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.histplot(data=dados, x='address', bins=30, kde=True)
    ax.set_title('Tempo (em anos) que a pessoa mora no mesmo endereço')
    st.pyplot(fig)

# educação (inteiros)
st.subheader('Nível de Educação')

fig, ax = plt.subplots()
sns.histplot(data=dados_filtrados, x='ed', bins=5, ax=ax)
ax.set_title('Distribuição - Nível de Educação')
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
st.pyplot(fig)

# matriz de correlação
st.subheader('Matriz de Correlação')
fig, ax = plt.subplots(figsize=(8,5))
sns.heatmap(dados_filtrados.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)