import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample # Adicionado para balanceamento

class BankLoanPipeline:
    def __init__(self, target='default', test_size=0.2, random_state=42, balance=True):
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.balance = balance # Novo parâmetro no init
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None

    def _validate_data(self, df):
        expected_debtinc = ((df['creddebt'] + df['othdebt']) / df['income']) * 100
        clean_df = df[np.abs(df['debtinc'] - expected_debtinc) < 0.5].copy()
        return clean_df.dropna()

    def _engineer_features(self, df):
        df = df.copy()
        df['total_debt'] = df['creddebt'] + df['othdebt']
        df['age'] = pd.to_numeric(df['age'], downcast='integer')
        df['ed'] = pd.to_numeric(df['ed'], downcast='integer')
        df = pd.get_dummies(df, columns=['ed'], prefix='edu_level')
        return df

    def _balance_data(self, X, y):
        """Privado: Realiza o Under-Sampling e reporta o status."""
        # Contagem antes do balanceamento
        antes = y.value_counts().to_dict()
        
        df_temp = pd.concat([X, y], axis=1)
        
        majoritaria = df_temp[df_temp[self.target] == 0]
        minoritaria = df_temp[df_temp[self.target] == 1]
        
        # Downsample
        majoritaria_downsampled = resample(
            majoritaria,
            replace=False, 
            n_samples=len(minoritaria),
            random_state=self.random_state
        )
        
        df_balanced = pd.concat([majoritaria_downsampled, minoritaria])
        df_balanced = df_balanced.sample(frac=1, random_state=self.random_state)
        
        X_bal, y_bal = df_balanced.drop(self.target, axis=1), df_balanced[self.target]
        
        # Relatório de Execução
        depois = y_bal.value_counts().to_dict()
        print("-" * 30)
        print("LOG DE BALANCEAMENTO (Under-Sampling)")
        print(f"Antes:  {antes}")
        print(f"Depois: {depois}")
        print("-" * 30)
        
        return X_bal, y_bal

    def process(self, df, training=True):
        df = self._validate_data(df)
        df = self._engineer_features(df)

        X = df.drop(self.target, axis=1) if self.target in df.columns else df
        y = df[self.target] if self.target in df.columns else None

        if training:
            # 1. Split Estratificado Inicial
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size,
                stratify=y, random_state=self.random_state
            )

            # 2. Balanceamento (Apenas nos dados de TREINO)
            if self.balance:
                X_train, y_train = self._balance_data(X_train, y_train)

            # 3. Escalonamento
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.is_fitted = True
            self.feature_names = X.columns.tolist()

            return X_train_scaled, X_test_scaled, y_train, y_test
        else:
            if not self.is_fitted:
                raise Exception("O pipeline precisa ser treinado antes.")
            return self.scaler.transform(X)

    def save_pipeline(self, filename='loan_pipeline.joblib'):
        """Salva o objeto inteiro (incluindo o scaler ajustado)."""
        joblib.dump(self, filename)
        print(f"Pipeline salvo com sucesso em: {filename}")
        
    @staticmethod
    def load_pipeline(filename='loan_pipeline.joblib'):
        """Método estático para carregar o pipeline salvo."""
        return joblib.load(filename) 