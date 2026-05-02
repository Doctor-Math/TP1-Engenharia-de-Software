import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class DataExplorer:
    """Classe responsável por gerar insights visuais sobre a base de dados."""
    
    def __init__(self, df):
        self.df = df

    def get_correlation_matrix(self):
        """Atende à necessidade de visualizar relações entre variáveis."""
        return self.df.corr()

    def plot_distribution(self, column, color='purple'):
        """Gera histogramas dinâmicos para o Streamlit."""
        fig, ax = plt.subplots()
        sns.histplot(data=self.df, x=column, kde=True, color=color, ax=ax)
        ax.set_title(f'Distribuição de {column}')
        return fig

    def get_missing_data_report(self):
        """Retorna estatísticas sobre dados faltantes."""
        return self.df.isna().sum()