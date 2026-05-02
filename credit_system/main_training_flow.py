import pandas as pd
import os
import joblib
import glob
from pathlib import Path # Forma mais moderna de lidar com caminhos
from sklearn.model_selection import train_test_split

# Importando suas classes
from data_engine import BankLoanPipeline
from models_engine import GradientBoostingModel, LogisticRegressionModel, ClusterModel

def main():
    # --- 1. GARANTIR A PASTA DE MODELOS ---
    # Detecta onde o script está e garante a pasta models lá dentro
    BASE_DIR = Path(__file__).resolve().parent
    PROJ_ROOT = BASE_DIR.parent
    MODELS_DIR = BASE_DIR / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Busca o arquivo bankloans.csv em qualquer subpasta do projeto
    search_pattern = str(PROJ_ROOT / "**" / "bankloans.csv")
    files = glob.glob(search_pattern, recursive=True)

    if not files:
        print(f"❌ ERRO: O arquivo 'bankloans.csv' sumiu! Procurei em: {PROJ_ROOT}")
        return

    DATA_PATH = files[0] # Pega o primeiro que encontrar
    print(f"✅ Arquivo encontrado em: {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH).dropna()

    # --- 3. PIPELINE ---

    pipeline = BankLoanPipeline(balance=True)
    # 1. Separar X e y antes do processamento
    X_raw = df.drop(columns=['default']) # ou o nome da sua coluna alvo
    y = df['default']

    # 2. O process agora retorna apenas os dados transformados
    X_processed = pipeline.process(X_raw, training=True)

    # 3. Fazemos a divisão manualmente aqui no main
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 4. TREINAMENTO E SALVAMENTO ---
    
    # Gradient Boosting
    print("Treinando Gradient Boosting...")
    model_gb = GradientBoostingModel(n_estimators=100, learning_rate=0.1)
    model_gb.train(X_train, y_train)
    model_gb.save(str(MODELS_DIR / 'model_final_gb.joblib'))

    # Regressão Logística
    cols_selecionadas = ['employ', 'address', 'debtinc', 'creddebt', 'edu_level_2']
    print(f"Treinando Regressão Logística...")
    model_lr = LogisticRegressionModel(selected_features=cols_selecionadas)
    model_lr.train(X_train, y_train, feature_names=pipeline.feature_names)
    model_lr.save(str(MODELS_DIR / 'model_final_lr.joblib'))


    # Clusterização
    print("Treinando Clusterização...")
    model_km = ClusterModel(n_clusters=4)
    model_km.train(X_train) 
    model_km.save(str(MODELS_DIR / 'model_clusters.joblib'))

    # Pipeline
    pipeline.save_pipeline(str(MODELS_DIR / 'loan_pipeline.joblib'))


    print("\n" + "="*30)
    print("SUCESSO: Todos os arquivos estão em /models")
    print("="*30)

# Este bloco garante que a função main() só rode se o script for executado diretamente
if __name__ == "__main__":
    main()