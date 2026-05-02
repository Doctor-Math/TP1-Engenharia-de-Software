import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#Titulo
st.title('Análise Variáveis de Inadimplência')

#----------------------------
# CARREGA DADOS
# ----------------------------
url = 'https://raw.githubusercontent.com/Doctor-Math/TP1-Engenharia-de-Software/main/credit_system/data/bankloans.csv'
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
dados['probabilidade'] = rg_model.predict_proba(X_scaled)[:, 1]

#----------------------------
# PAINEL DE DEFINIÇÃO DA PROBABILIDADE
# ----------------------------
st.sidebar.header("Limiar de probabilidade de Inadimplência")
limiar = st.sidebar.slider("Limiar:", 0.0, 1.0, 0.5, 0.1)

# Classificação
dados['status'] = dados['probabilidade'].apply(
    lambda x: 'Negado' if x >= limiar else 'Aprovado'
)

#----------------------------
# PAINEL GERAL
# ----------------------------
c1, c2, = st.columns(2)

#conta quantos clientes foram aprovados
aprovados = len(dados[dados['status'] == 'Aprovado'])
c1.metric("Total de Clientes", len(dados))
c2.metric("Clientes Aprovados",aprovados)

st.divider()

#----------------------------
# MAPA DE RISCO
# ----------------------------
st.subheader("Mapa de Risco")
fig_disp, ax_disp = plt.subplots(figsize=(10, 5))

#Plota dispersão da probabilidade em relaçao debtin
scatter = ax_disp.scatter(data=dados, x='debtinc', y='probabilidade', c='probabilidade')
#Plota linha do limiar de probabilidade
ax_disp.axhline(limiar, color='black', linestyle='--', label=f'Corte: {limiar}')

#Legendas
plt.colorbar(scatter, label='Risco')
ax_disp.set_xlabel("Dívida/Renda")
ax_disp.set_ylabel("Probabilidade de Risco")
st.pyplot(fig_disp)

st.divider()

#----------------------------
# MÉDIA POR STATUS
# ----------------------------
st.subheader("Perfil Médio: Aprovados e Negados")

#Agrupa em relação ao status do cliente (aprovado, negado)
df_medias = dados.groupby('status')[features].mean().T
df_medias.columns = ["Aprovado", "Negado"]

#Plota as colunas para média da variavel cada status
fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
df_medias.plot(kind='bar', ax=ax_bar, color=['#2ca02c', '#d62728'], rot=0)
ax_bar.bar_label(ax_bar.containers[0], fmt='%.1f', padding=3)
ax_bar.bar_label(ax_bar.containers[1], fmt='%.1f', padding=3)

st.pyplot(fig_bar)