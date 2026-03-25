# Análise de Crédito

![Jupyter Notebook](https://img.shields.io/badge/Notebook-Jupyter-orange)

## Objetivos e Principais Features

O objetivo deste projeto é desenvolver um modelo de análise de crédito capaz de prever a probabilidade de inadimplência de clientes com base em dados históricos de crédito. A proposta é auxiliar instituições financeiras na tomada de decisão, reduzindo riscos e aumentando a eficiência na concessão de crédito. Para isto, disporemos da base [Credit Risk Analysis for extending Bank Loans](https://www.kaggle.com/datasets/atulmittal199174/credit-risk-analysis-for-extending-bank-loans), do Kaggle, composta por uma amostra de 1000 clientes com 9 features: idade, nível de educação, experiência de trabalho, endereço, renda anual, índice de endividamento, relação crédito/renda, outras dívidas e histórico de inadimplência.

O sistema irá realizar o pré-processamento dos dados, incluindo tratamento de valores ausentes, normalização e análise exploratória. Em seguida, serão aplicados algoritmos de aprendizado de máquina para classificação dos clientes em perfis de risco. Entre as principais atividades estão: análise exploratória dos dados (EDA), engenharia de atributos, treinamento e avaliação de modelos (como regressão logística e árvores de decisão), comparação de desempenho entre modelos e visualização dos resultados. Também será possível interpretar os fatores que mais influenciam o risco de crédito, promovendo maior transparência no processo decisório.

Dessa forma, o projeto busca atender diretamente a demandas reais do mercado financeiro, como a necessidade de responder perguntas fundamentais no processo de concessão de crédito: dado o conjunto de características de um cliente, ele é confiável ou apresenta alto risco de inadimplência? A partir dessa previsão, o objetivo do modelo é permitir automatizar e padronizar decisões que tradicionalmente dependiam de análises manuais, tornando o processo mais ágil e escalável.

Outra demanda importante contemplada é a compreensão dos fatores que levam à inadimplência. Ou seja, além de classificar o cliente, o sistema também possibilita identificar quais características são mais frequentes entre os clientes que não pagam seus empréstimos, como altos níveis de endividamento, baixa renda ou histórico negativo. Essa análise fornece insights estratégicos para as instituições, permitindo não apenas mitigar riscos, mas também ajustar políticas de crédito, definir limites mais adequados e até criar estratégias para diferentes perfis de clientes.

Assim, o projeto não só prevê o risco, mas também gera conhecimento útil para tomada de decisão, alinhando-se à crescente demanda do mercado por soluções baseadas em dados, explicáveis e orientadas à redução de risco.

## Membros da Equipe e Papéis

*   [Anny Caroline Almida Marcelino](https://github.com/AnnyACAM): Cientista de Dados / Machine Learning (modelagem e avaliação dos algoritmos)
*   [Carolina Penido Barcellos](https://github.com/carolinabarcellos): Analista de Dados (visualização de dados e interpretação dos resultados)
*   [Gabrielly Xavier dos Santos](https://github.com/gabyxsantos): Cientista de Dados (análise exploratória e pré-processamento dos dados)
*   [Matheus Soares dos Santos de Freitas](https://github.com/Doctor-Math): Engenheiro de Dados (tratamento, organização e preparação do dataset)


## Pilha de Tecnologias

### 💻 Ambiente
- Google Colab

### 🐍 Linguagem
- Python 3

### ⚙️ Hardware
- CPU  
- GPU (NVIDIA T4, disponível quando necessário)

### 📚 Bibliotecas
- Pandas  
- NumPy  
- Scikit-learn  
- Seaborn  
- Matplotlib  
