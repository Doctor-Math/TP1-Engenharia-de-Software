import pandas as pd
import numpy as np
import os
from data_engine import BankLoanPipeline  # <-- Importante: traz a lógica do outro arquivo

def main():
    # URL da base de dados
    url = 'https://raw.githubusercontent.com/Doctor-Math/TP1-Engenharia-de-Software/main/data/bankloans.csv'
    
    print("Carregando dados...")
    df_bank_loans = pd.read_csv(url)

    # Instanciar o Pipeline (que veio do data_engine)
    pipeline = BankLoanPipeline()

    print("Processando e treinando o pipeline...")
    X_train, X_test, y_train, y_test = pipeline.process(df_bank_loans)

    # Nome da pasta para salvar
    output_dir = 'data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Salva o pipeline e os dados na pasta correta
    pipeline.save_pipeline(os.path.join(output_dir, 'pipeline_tratamento_bankloans.joblib'))
    
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'Y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'Y_test.npy'), y_test)
    
    print(f"Treinamento concluído e arquivos salvos em: {output_dir}/")

if __name__ == "__main__":
    main()