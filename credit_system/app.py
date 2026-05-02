import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Importando suas classes customizadas
from data_engine import BankLoanPipeline
from models_engine import GradientBoostingModel, ClusterModel
from explorer import DataExplorer
from evaluator import BiasEvaluator

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Sistema de Análise de Crédito", layout="wide")

# 1. Ajuste de Caminhos (Baseado na localização deste app.py)
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "bankloans.csv" # Caminho seguro para o CSV

@st.cache_resource
def load_assets():
    """Carrega pipeline e modelos com tratamento de erro detalhado."""
    pipeline_path = MODELS_DIR / "loan_pipeline.joblib"
    gb_path = MODELS_DIR / "model_final_gb.joblib"
    cluster_path = MODELS_DIR / "model_clusters.joblib"

    # Verificação individual para saber qual arquivo falta
    for p in [pipeline_path, gb_path, cluster_path]:
        if not p.exists():
            st.error(f"Arquivo não encontrado: {p.name} em {MODELS_DIR}")
            st.stop()

    # Carregamento usando strings (evita erro no joblib com objetos Path)
    pipeline = BankLoanPipeline.load_pipeline(str(pipeline_path))
    model_credit = GradientBoostingModel.load(str(gb_path))
    model_cluster = ClusterModel.load(str(cluster_path))
    
    return pipeline, model_credit, model_cluster

# Chamada direta (sem o try/except genérico para podermos ver o erro real se ocorrer)
pipeline, model_credit, model_cluster = load_assets()

# --- CARREGAMENTO DA BASE ---
# Usando o DATA_PATH que definimos acima
if DATA_PATH.exists():
    df_raw = pd.read_csv(DATA_PATH)
else:
    st.error(f"Base de dados não encontrada em: {DATA_PATH}")
    st.stop()

# --- SIDEBAR / MENU ---
st.sidebar.title("Menu de Navegação")
aba = st.sidebar.radio("Selecione uma visão:", 
    ["Exploração de Dados (EDA)", "Análise de Crédito", "Segmentação (Marketing)", "Auditoria de Ética"])

# --- CARREGAMENTO DA BASE ---

# Opção A: data está na raiz (fora de credit_system)
path_externo = BASE_DIR.parent / "data" / "bankloans.csv"
# Opção B: data está dentro de credit_system
path_interno = BASE_DIR / "data" / "bankloans.csv"

if path_externo.exists():
    df_raw = pd.read_csv(path_externo)
elif path_interno.exists():
    df_raw = pd.read_csv(path_interno)
else:
    st.error(f"❌ Erro Crítico: O arquivo 'bankloans.csv' não foi encontrado.")
    st.info(f"Verifique se o arquivo está em: {path_externo} OU {path_interno}")
    st.stop()

# --- LÓGICA DAS ABAS ---

if aba == "Exploração de Dados (EDA)":
    st.title("🔍 Análise Exploratória de Dados")
    explorer = DataExplorer(df_raw)
    
    col1, col2 = st.columns(2)
    with col1:
        var = st.selectbox("Escolha uma variável para ver a distribuição:", df_raw.columns)
        st.pyplot(explorer.plot_distribution(var))
    with col2:
        st.write("### Matriz de Correlação")
        st.dataframe(explorer.get_correlation_matrix().style.background_gradient(cmap='coolwarm'))

elif aba == "Análise de Crédito":
    st.title("💳 Simulador de Aprovação de Crédito")
    st.info("Preencha os dados do cliente para obter a predição em tempo real.")
    
    with st.form("form_cliente"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Idade", 18, 100, 30)
        income = c2.number_input("Renda Anual ($)", 1000, 1000000, 50000)
        employ = c3.number_input("Anos no emprego atual", 0, 50, 5)
        
        c4, c5, c6 = st.columns(3)
        debtinc = c4.slider("Debt-to-Income Ratio", 0.0, 50.0, 10.0)
        creddebt = c5.number_input("Dívida de Cartão", 0.0, 100000.0, 1000.0)
        othdebt = c6.number_input("Outras Dívidas", 0.0, 100000.0, 2000.0)
        
        submit = st.form_submit_button("Analisar Crédito")

    if submit:
        # 1. Dados brutos conforme o CSV original
        dados_usuario = {
            'age': age,
            'ed': 1, # Vamos usar 1 como padrão, mas o código abaixo resolve o erro
            'employ': employ,
            'address': 5,
            'income': income,
            'debtinc': debtinc,
            'creddebt': creddebt,
            'othdebt': othdebt
        }
        
        input_data = pd.DataFrame([dados_usuario])
        
        try:
            # 2. Processar via Pipeline
            X_processed = pipeline.process(input_data, training=False)
            
            # --- O PULO DO GATO ---
            # Transformamos em DataFrame para garantir que todas as colunas do treino existam
            X_df = pd.DataFrame(X_processed, columns=pipeline.feature_names)
            
            # Se faltar alguma coluna (como as de educação), o código abaixo preenche com 0
            for col in pipeline.feature_names:
                if col not in X_df.columns:
                    X_df[col] = 0
            
            # Reordenamos para ficar igual ao fit do modelo
            X_final = X_df[pipeline.feature_names].values
            # ----------------------

            # 3. Predição usando os dados alinhados
            pred = model_credit.predict(X_final)
            
            st.divider()
            if pred[0] == 0:
                st.balloons()
                st.success("## ✅ Crédito Aprovado!")
            else:
                st.error("## ❌ Crédito Negado")
                
        except Exception as e:
            st.error(f"Erro técnico: {e}")

elif aba == "Segmentação (Marketing)":
    st.title("🎯 Segmentação de Perfil de Cliente")
    st.markdown("Esta visualização utiliza **K-Means** para agrupar clientes e **PCA** para reduzir a dimensionalidade.")

    # 1. Processar a base inteira para o cluster
    # Removemos o alvo 'default' se ele existir no df_raw para não enviesar o cluster
    X_raw = df_raw.drop(columns=['default'], errors='ignore')
    X_processed = pipeline.process(X_raw, training=False)
    
    # 2. Obter Clusters e Coordenadas PCA
    clusters = model_cluster.predict(X_processed)
    pca_coords = model_cluster.get_pca_coords(X_processed)
    
    # 3. Plotar
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(pca_coords[:, 0], pca_coords[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    ax.set_title("Clusters de Clientes (Visualização 2D via PCA)")
    ax.set_xlabel("Componente Principal 1")
    ax.set_ylabel("Componente Principal 2")
    plt.colorbar(scatter, label='Cluster ID')
    
    st.pyplot(fig)
    
    # 4. Insights dos Clusters
    st.write("### 📊 Perfil dos Grupos")
    df_with_clusters = df_raw.copy()
    df_with_clusters['Cluster'] = clusters
    st.dataframe(df_with_clusters.groupby('Cluster').mean())

elif aba == "Auditoria de Ética":
    st.title("⚖️ Auditoria de Viés Algorítmico")
    st.warning("Análise de paridade preditiva: verificamos se o modelo comete mais erros de 'Falso Negativo' em certos grupos.")
    
    # 1. Preparar dados para auditoria
    X_raw = df_raw.drop(columns=['default'], errors='ignore')
    y_true = df_raw['default']
    
    X_processed = pipeline.process(X_raw, training=False)
    # Alinhamento de colunas (mesma lógica do simulador)
    X_df = pd.DataFrame(X_processed, columns=pipeline.feature_names).fillna(0)
    y_pred = model_credit.predict(X_df[pipeline.feature_names].values)
    
    # 2. Usar o BiasEvaluator (Sua classe customizada)
    evaluator = BiasEvaluator()
    
    # Exemplo: Auditoria por nível de renda (Income)
    # Vamos criar faixas de renda para auditar
    df_audit = df_raw.copy()
    df_audit['y_true'] = y_true
    df_audit['y_pred'] = y_pred
    df_audit['faixa_renda'] = pd.qcut(df_audit['income'], q=3, labels=['Baixa', 'Média', 'Alta'])
    
    st.write("### Taxa de Falso Positivo (FPR) por Faixa de Renda")
    st.info("FPR alto significa que o modelo tende a negar crédito para bons pagadores deste grupo com mais frequência.")
    
    # Cálculo simplificado de viés
    bias_report = []
    for grupo in df_audit['faixa_renda'].unique():
        subset = df_audit[df_audit['faixa_renda'] == grupo]
        # FPR = FP / (FP + TN) -> Onde y_true era 0 mas predizemos 1
        fp = ((subset['y_true'] == 0) & (subset['y_pred'] == 1)).sum()
        tn = ((subset['y_true'] == 0) & (subset['y_pred'] == 0)).sum()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        bias_report.append({'Faixa de Renda': grupo, 'FPR (Risco de Injustiça)': f"{fpr:.2%}"})
    
    st.table(pd.DataFrame(bias_report))