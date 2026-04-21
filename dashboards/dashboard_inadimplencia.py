import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#Titulo
st.title('Análise Variáveis de Inadimplência')

#----------------------------
# CARREGA DADOS
# ----------------------------
url = 'https://raw.githubusercontent.com/Doctor-Math/TP1-Engenharia-de-Software/main/data/bankloans.csv'
dados = pd.read_csv(url).dropna()
dados = pd.get_dummies(dados, columns=['ed'], prefix='edu_level')
features = ["employ", "address", "debtinc", "creddebt", "edu_level_2"] # variaveis que estimam o modelo

#----------------------------
# NORMALIZA
# ----------------------------
y = dados['default']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(dados[features])

#----------------------------
# MODELO
# ----------------------------
rg_model = LogisticRegression()
rg_model.fit(X_scaled, y)

#----------------------------
# FILTRO
# ----------------------------
st.sidebar.header('Filtro de Variável')
variavel = st.sidebar.selectbox('Analise de:', features)

#----------------------------
# PAINEL MODELO
# ----------------------------
with st.container():
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Variáveis Originais", "13")
    with c2:
        st.metric("Variáveis Otimizadas", len(features))
    
st.divider()

#Métricas retirada do modelo de inadimplencia
st.subheader("Performance do Modelo")
m1, m2, m3, m4 = st.columns(4)
m1.metric("AUC-ROC", "0.85", help="Capacidade global do modelo de distinguir entre um 'bom' e um 'mau' pagador")
m2.metric("F1-Score", "0.65", help="Balanço entre Precisão e Recall")
m3.metric("Recall", "0.78", help="Proporção de inadimplentes reais que o modelo detectou")
m4.metric("Precision", "0.56", help="Assertividade do modelo")

# Matriz de Correlação
st.write("Matriz de Correlação")
fig_corr, ax_corr = plt.subplots(figsize=(8, 4))
corr_matrix = dados[features + ['default']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, ax=ax_corr, fmt=".2f")
st.pyplot(fig_corr)

st.divider()

#----------------------------
# RISCO POR VARIÁVEL
# ----------------------------
# Inadimplentes por faixa de cada variavel
st.subheader(f"Risco por Faixa: {variavel}")

#Separa a variavel em 5 intervalos
dados['faixa'] = pd.qcut(dados[variavel], q=5, duplicates='drop').astype(str)
#Calcula a média de inadimplentes nesse intervalo
var_group_taxa = dados.groupby('faixa')['default'].mean().reset_index()

#Plota
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(data=var_group_taxa, x='faixa', y='default', palette='OrRd', ax=ax)
ax.set_ylabel("Taxa de Inadimplência (%)")
ax.set_xlabel(f"Intervalos de {variavel}")
st.pyplot(fig)

#----------------------------
# RELEVÂNCIA DA VARIÁVEL
# ----------------------------
st.subheader("Peso da variável no Modelo Reduzido")

#DataFrame com o peso de cada variavel para o modelo de regressão logistica
df_coef = pd.DataFrame({'Variável': features,'Peso': rg_model.coef_[0]}).sort_values(by='Peso', ascending=False)

#Plota destacando a variavel selecionada
fig2, ax2 = plt.subplots(figsize=(10, 3))
cores = ['#d9534f' if v == variavel else '#5bc0de' for v in df_coef['Variável']]
sns.barplot(data=df_coef, x='Peso', y='Variável', palette=cores, ax=ax2)
ax2.axvline(0, color='black', ls='--')
st.pyplot(fig2)