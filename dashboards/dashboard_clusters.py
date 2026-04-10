import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import plotly.express as px

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Segmentação de Clientes", layout="wide")

st.title("Dashboard de Segmentação de Clientes")

# =========================
# UPLOAD
# =========================
file = "bankloans.csv"

df = pd.read_csv(file)
df = df.dropna()

# =========================
# FEATURES
# =========================
features = [
    'age', 'ed', 'employ', 'address',
     'income', 'debtinc', 'creddebt',
    'othdebt']

X = df[features]

# =========================
# NORMALIZAÇÃO
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# ESCOLHA DE K
# =========================
st.sidebar.title("⚙️ Configurações")
k = st.sidebar.slider("Número de Clusters (k)", 2, 8, 4)

# =========================
# MODELO
# =========================
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# =========================
# PCA
# =========================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['pca1'] = X_pca[:, 0]
df['pca2'] = X_pca[:, 1]

# =========================
# VISUALIZAÇÃO
# =========================
st.subheader("Visualização dos Clusters")

fig = px.scatter(
    df,
    x='pca1',
    y='pca2',
    color=df['cluster'].astype(str),
    title="Clusters de Clientes",
    hover_data=features
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# PERFIL DOS CLUSTERS
# =========================
st.subheader("Perfil dos Clusters")

profile = df.groupby('cluster')[features].mean()
st.dataframe(profile)

# =========================
# ESTRATÉGIAS
# =========================
st.subheader("Estratégia por Cluster")

def estrategia(row):
    if row['debtinc'] > 15 or row['othdebt'] >5 or row['creddebt']>10:
        return "Alto risco - restringir crédito"
    elif row['income'] > 50 and row['debtinc'] < 10:
        return "Cliente premium - oferecer produtos exclusivos"
    elif row['income'] < 40:
        return "Crescimento - oferecer crédito inicial"
    else:
        return "Moderado - crédito com cautela"

profile['estrategia'] = profile.apply(estrategia, axis=1)

st.dataframe(profile[['estrategia']])

# =========================
# DISTRIBUIÇÃO
# =========================
st.subheader("Distribuição de Clientes")

fig2 = px.histogram(df, x='cluster', title="Quantidade por Cluster")
st.plotly_chart(fig2, use_container_width=True)
