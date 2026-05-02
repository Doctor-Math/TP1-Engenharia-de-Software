import pandas as pd
from sklearn.metrics import confusion_matrix

class BiasEvaluator:
    """Classe responsável por auditar a justiça (fairness) do modelo."""
    
    @staticmethod
    def calculate_fpr(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        # TN, FP, FN, TP
        if cm.size == 4:
            TN, FP, FN, TP = cm.ravel()
            return FP / (FP + TN) if (FP + TN) > 0 else 0
        return 0

    def audit_bias(self, df_test, protected_feature, target='y_true', prediction='y_pred'):
        """Gera um relatório de disparidade por grupo."""
        results = []
        for group in df_test[protected_feature].unique():
            df_group = df_test[df_test[protected_feature] == group]
            fpr = self.calculate_fpr(df_group[target], df_group[prediction])
            
            results.append({
                'Group': group,
                'FPR': fpr,
                'Sample Size': len(df_group)
            })
        
        return pd.DataFrame(results).sort_values(by='FPR', ascending=False)