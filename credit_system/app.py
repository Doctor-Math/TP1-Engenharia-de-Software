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
    ["Exploração de Dados (EDA)", "Análise de Crédito", "Segmentação (Marketing)", "Fatores de Inadimplência", "Auditoria de Ética"])

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
    
    # --- FERRAMENTA: AJUSTE DE LIMIAR ---
    st.sidebar.divider()
    st.sidebar.subheader("⚙️ Configurações de Risco")
    limiar_corte = st.sidebar.slider(
        "Limiar de Aceitação (Cut-off):", 
        min_value=0.0, max_value=1.0, value=0.50, step=0.05,
        help="Quanto menor o limiar, mais rígida é a aprovação. Padrão: 0.50"
    )

    st.info("Preencha os dados do cliente para obter a predição e segmentação em tempo real.")
    
    with st.form("form_cliente"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Idade", 18, 100, 30)
        income = c2.number_input("Renda Anual ($)", 0, 1000000, 50000)
        employ = c3.number_input("Anos no emprego atual", 0, 50, 5)
        
        c4, c5, c6 = st.columns(3)
        creddebt = c4.number_input("Dívida de Cartão ($)", 0.0, 100000.0, 1000.0)
        othdebt = c5.number_input("Outras Dívidas ($)", 0.0, 100000.0, 2000.0)
        ed_level = c6.number_input("Nível educacional (1 a 4):", 1, 4, 1)
        
        if income > 0:
            debtinc_calculado = ((creddebt + othdebt) / income) * 100
        else:
            debtinc_calculado = 0.0
            
        st.write(f"**Razão Dívida/Renda Calculada:** {debtinc_calculado:.2f}%")
        submit = st.form_submit_button("Analisar Crédito")

    if submit:
        # 1. Preparação dos dados (Mesma escala do modelo)
        income_mod = income / 1000
        cred_mod = creddebt / 1000
        oth_mod = othdebt / 1000
        debtinc_final = ((cred_mod + oth_mod) / income_mod * 100) if income_mod > 0 else 0
        
        dados_usuario = {
            'age': age, 'ed': ed_level, 'employ': employ, 'address': 5, 
            'income': income_mod, 'debtinc': debtinc_final, 
            'creddebt': cred_mod, 'othdebt': oth_mod
        }
        
        input_data = pd.DataFrame([dados_usuario])
        
        try:
            # 2. Processamento
            X_processed = pipeline.process(input_data, training=False)
            X_df = pd.DataFrame(X_processed, columns=pipeline.feature_names).fillna(0)
            X_final = X_df[pipeline.feature_names].values

            # --- FERRAMENTA: PREDIÇÃO COM LIMIAR PERSONALIZADO ---
            # Pegamos a probabilidade da classe 1 (Inadimplente)
            prob_inadimplencia = model_credit.model.predict_proba(X_final)[0][1]

            st.subheader("🎯 Resultado da Análise")
            # Se a prob for maior que o limiar escolhido, negamos
            if prob_inadimplencia >= limiar_corte:
                st.error(f"### ❌ Crédito Negado")
                st.write(f"Risco Calculado: **{prob_inadimplencia:.1%}**")
                st.write(f"Limiar de Corte: **{limiar_corte:.1%}**")
            else:
                st.balloons()
                st.success(f"### ✅ Crédito Aprovado!")
                st.write(f"Risco Calculado: **{prob_inadimplencia:.1%}**")

            # Barra visual de risco
            st.progress(prob_inadimplencia)

        except Exception as e:
            st.error(f"Erro técnico na predição: {e}")

elif aba == "Segmentação (Marketing)":
    st.title("🎯 Dashboard de Segmentação de Clientes")
    st.markdown("Esta visão utiliza **IA** para agrupar clientes com comportamentos financeiros semelhantes.")

    import plotly.express as px

    # 1. Processamento dos Dados
    # Usamos o pipeline para garantir que a escala seja a mesma do treino
    X_raw = df_raw.drop(columns=['default'], errors='ignore')
    X_processed = pipeline.process(X_raw, training=False)
    
    # 2. Predição de Clusters e PCA
    # Usamos o seu modelo carregado para manter a consistência
    clusters = model_cluster.predict(X_processed)
    pca_coords = model_cluster.get_pca_coords(X_processed)
    
    # Adicionamos os resultados ao DataFrame original para visualização
    df_viz = df_raw.copy()
    df_viz['cluster'] = clusters.astype(str) # Transformar em string para o Plotly tratar como categórico
    df_viz['pca1'] = pca_coords[:, 0]
    df_viz['pca2'] = pca_coords[:, 1]

    # 3. Visualização Interativa (Plotly)
    col_graph, col_dist = st.columns([2, 1])

    with col_graph:
        st.subheader("Mapa Espacial de Clusters")
        fig_scatter = px.scatter(
            df_viz,
            x='pca1',
            y='pca2',
            color='cluster',
            title="Distribuição de Clientes (Redução de Dimensionalidade PCA)",
            hover_data=['age', 'income', 'employ', 'debtinc'],
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_dist:
        st.subheader("Volume por Grupo")
        fig_hist = px.histogram(df_viz, x='cluster', color='cluster', 
                               color_discrete_sequence=px.colors.qualitative.Safe)
        st.plotly_chart(fig_hist, use_container_width=True)

    # 4. Tabela de Perfil e Estratégia
    st.divider()
    st.subheader("📋 Perfil Médio e Estratégia de Negócio")

    # Calculamos a média por cluster
    features = ['age', 'income', 'employ', 'debtinc', 'creddebt', 'othdebt']
    profile = df_viz.groupby('cluster')[features].mean()

    # Função de Estratégia (Baseada no seu exemplo)
    def sugerir_estrategia(row):
        if row['debtinc'] > 15 or row['creddebt'] > 5:
            return "⚠️ Alto Risco: Restringir crédito e monitorar."
        elif row['income'] > 60 and row['debtinc'] < 10:
            return "💎 Premium: Oferecer cartões Black e investimentos."
        elif row['employ'] > 10:
            return "🛡️ Estável: Oferecer aumento de limite progressivo."
        else:
            return "🌱 Potencial: Crédito inicial com juros moderados."

    profile['Sugestão de Estratégia'] = profile.apply(sugerir_estrategia, axis=1)

    # Exibição estilizada
    st.dataframe(
        profile.style.background_gradient(cmap='Greens', subset=['income'])
        .format("{:.2f}", subset=features)
    )

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

elif aba == "Fatores de Inadimplência":
    st.title("📊 Análise das Variáveis de Inadimplência")
    st.markdown("Nesta seção, exploramos como cada variável impacta o risco de crédito.")

    # 1. Filtro de Variável (Sidebar específica para esta aba)
    features_analise = ["employ", "debtinc", "creddebt", "othdebt", "income", "age"]
    st.sidebar.divider()
    st.sidebar.subheader("🔍 Filtro de Análise")
    var_alvo = st.sidebar.selectbox('Analisar impacto de:', features_analise)

    # 2. Painel de Métricas do Modelo
    st.subheader("Performance Global do Modelo")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AUC-ROC", "0.85", help="Capacidade de distinguir bons e maus pagadores.")
    m2.metric("F1-Score", "0.65")
    m3.metric("Recall", "0.78")
    m4.metric("Precision", "0.56")

    st.divider()

    col_graf, col_info = st.columns([2, 1])

    with col_graf:
        # 3. Risco por Faixa (Gráfico de Barras)
        st.subheader(f"Taxa de Inadimplência por Faixa: {var_alvo}")
        
        # Criamos as faixas de dados para a variável selecionada
        df_plot = df_raw.copy()
        df_plot['faixa'] = pd.qcut(df_plot[var_alvo], q=5, duplicates='drop').astype(str)
        var_group = df_plot.groupby('faixa')['default'].mean().reset_index()
        
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=var_group, x='faixa', y='default', palette='Reds', ax=ax)
        ax.set_ylabel("Inadimplência Média (%)")
        st.pyplot(fig)

    with col_info:
        # 4. Peso das Variáveis (Feature Importance)
        st.subheader("Importância no Modelo")
        # Pegamos a importância do seu modelo de Gradient Boosting
        importances = model_credit.model.feature_importances_
        # Mapeamos para as colunas do pipeline
        feat_imp = pd.DataFrame({'Variável': pipeline.feature_names, 'Importância': importances})
        feat_imp = feat_imp.sort_values(by='Importância', ascending=True).tail(8)

        fig_imp, ax_imp = plt.subplots()
        # Destacamos a variável que o usuário selecionou
        colors = ['#d9534f' if v in var_alvo else '#5bc0de' for v in feat_imp['Variável']]
        ax_imp.barh(feat_imp['Variável'], feat_imp['Importância'], color=colors)
        st.pyplot(fig_imp)

    st.divider()

    # 5. Mapa de Risco Dinâmico
    st.subheader("Simulador de Limiar de Risco (Cut-off)")
    limiar = st.slider("Ajuste o limite de tolerância ao risco:", 0.0, 1.0, 0.5)

    # Pegamos as probabilidades do modelo
    X_proc = pipeline.process(df_raw.drop(columns=['default']), training=False)
    probs = model_credit.model.predict_proba(X_proc)[:, 1]
    
    df_raw['probabilidade'] = probs
    df_raw['decisao'] = df_raw['probabilidade'].apply(lambda x: 'Negado' if x >= limiar else 'Aprovado')

    c1, c2 = st.columns(2)
    aprovados = (df_raw['decisao'] == 'Aprovado').sum()
    c1.metric("Clientes Aprovados", aprovados, f"{aprovados/len(df_raw):.1%}")
    c2.metric("Clientes Negados", len(df_raw)-aprovados, f"{(len(df_raw)-aprovados)/len(df_raw):.1%}")

    # Gráfico de dispersão do Risco
    import plotly.express as px
    fig_risk = px.scatter(df_raw, x='debtinc', y='probabilidade', color='decisao',
                         title="Probabilidade de Risco vs. Endividamento",
                         labels={'debtinc': 'Relação Dívida/Renda', 'probabilidade': 'Risco Calculado'},
                         color_discrete_map={'Aprovado': '#2ca02c', 'Negado': '#d62728'})
    fig_risk.add_hline(y=limiar, line_dash="dash", line_color="black", annotation_text="Limiar de Corte")
    st.plotly_chart(fig_risk, use_container_width=True)
