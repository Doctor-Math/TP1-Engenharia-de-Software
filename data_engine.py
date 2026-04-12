import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class BankLoanPipeline:
    def __init__(self, target='default', test_size=0.2, random_state=42):
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None

    def _validate_data(self, df):
        """Privado: Checa integridade matemática."""
        # Calculando a dívida esperada para validar os dados
        expected_debtinc = ((df['creddebt'] + df['othdebt']) / df['income']) * 100
        # Tolerância de 0.5 para arredondamentos
        clean_df = df[np.abs(df['debtinc'] - expected_debtinc) < 0.5].copy()
        return clean_df.dropna()

    def _engineer_features(self, df):
        """Privado: Cria novas colunas e otimiza tipos."""
        df = df.copy()
        df['total_debt'] = df['creddebt'] + df['othdebt']

        # Otimização de tipos para economia de memória
        df['age'] = pd.to_numeric(df['age'], downcast='integer')
        df['ed'] = pd.to_numeric(df['ed'], downcast='integer')

        # One-Hot Encoding para escolaridade
        df = pd.get_dummies(df, columns=['ed'], prefix='edu_level')
        return df

    def process(self, df, training=True):
        """Ponto de entrada principal do pipeline."""
        df = self._validate_data(df)
        df = self._engineer_features(df)

        X = df.drop(self.target, axis=1) if self.target in df.columns else df
        y = df[self.target] if self.target in df.columns else None

        if training:
            # Split Estratificado
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size,
                stratify=y, random_state=self.random_state
            )

            # Ajusta o scaler apenas nos dados de treino
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            self.is_fitted = True
            self.feature_names = X.columns.tolist()

            return X_train_scaled, X_test_scaled, y_train, y_test
        else:
            # Para inferência (novos dados), apenas transforma
            if not self.is_fitted:
                raise Exception("O pipeline precisa ser treinado antes de processar dados de teste.")
            return self.scaler.transform(X)

    def save_pipeline(self, filename='loan_pipeline.joblib'):
        """Salva o objeto inteiro (incluindo o scaler ajustado)."""
        joblib.dump(self, filename)
        print(f"Pipeline salvo com sucesso em: {filename}")

    @staticmethod
    def load_pipeline(filename='loan_pipeline.joblib'):
        """Método estático para carregar o pipeline salvo."""
        return joblib.load(filename)