import pandas as pd
import joblib
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

class BaseModel(ABC):
    def __init__(self, model_obj, selected_features=None):
        self.model = model_obj
        self.selected_features = selected_features 

    def _filter_features(self, X, feature_names):
        if self.selected_features and feature_names:
            df_temp = pd.DataFrame(X, columns=feature_names)
            return df_temp[self.selected_features].values
        return X

    def train(self, X, y, feature_names=None):
        X_filtered = self._filter_features(X, feature_names)
        self.model.fit(X_filtered, y)

    def predict(self, X, feature_names=None):
        X_filtered = self._filter_features(X, feature_names)
        return self.model.predict(X_filtered)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        report = classification_report(y, predictions, output_dict=True)
        matrix = confusion_matrix(y, predictions)
        return report, matrix

    def save(self, filename):
        # Salva o objeto da nossa classe inteiro (self), não só o self.model
        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename):
        # Carrega o objeto da nossa classe
        return joblib.load(filename)

# --- Implementações Específicas ---

class RandomForestModel(BaseModel):
    def __init__(self, model_obj=None, **kwargs):
        if model_obj is None:
            model_obj = RandomForestClassifier(**kwargs, random_state=42)
        super().__init__(model_obj)

class GradientBoostingModel(BaseModel):
    def __init__(self, model_obj=None, **kwargs):
        if model_obj is None:
            model_obj = GradientBoostingClassifier(**kwargs, random_state=42)
        super().__init__(model_obj)
    
    def get_feature_importance(self, feature_names):
        importances = self.model.feature_importances_
        return pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)

class SVMModel(BaseModel):
    def __init__(self, model_obj=None, **kwargs):
        if model_obj is None:
            model_obj = SVC(**kwargs, probability=True, random_state=42)
        super().__init__(model_obj)

class ClusterModel(BaseModel):
    def __init__(self, model_obj=None, n_clusters=4, **kwargs):
        if model_obj is None:
            model_obj = KMeans(n_clusters=n_clusters, n_init=10, random_state=42, **kwargs)
        super().__init__(model_obj)
        self.pca = PCA(n_components=2)

    def train(self, X, y=None):
        self.model.fit(X)
        # IMPORTANTE: Ajustar o PCA aos dados de treino
        self.pca.fit(X)

    def predict(self, X):
        return self.model.predict(X)

    def get_pca_coords(self, X):
        return self.pca.transform(X)
    
class LogisticRegressionModel(BaseModel):
    def __init__(self, model_obj=None, selected_features=None, **kwargs):
        if model_obj is None:
            model_obj = LogisticRegression(class_weight='balanced', **kwargs, random_state=42)
        super().__init__(model_obj, selected_features=selected_features)

class ModelManager:
    def __init__(self, pipeline, model):
        self.pipeline = pipeline  
        self.model = model        

    def run_inference(self, raw_data):
        processed_data = self.pipeline.process(raw_data, training=False)
        prediction = self.model.predict(processed_data)
        try:
            prob = self.model.model.predict_proba(processed_data)[:, 1]
            return prediction, prob
        except:
            return prediction, None