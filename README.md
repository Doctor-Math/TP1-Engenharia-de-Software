# Análise de Crédito

![Jupyter Notebook](https://img.shields.io/badge/Notebook-Jupyter-orange)

## 🎯 Objetivos e Principais Features
Este projeto se propõe a desenvolver um modelo de análise de risco para prever a probabilidade de inadimplência, auxiliando bancos e outras instituições financeiras na redução de perdas durante suas políticas de concessão de crédito a clientes. O foco central, portanto, é determinar a confiabilidade de usuários bancários por meio da identificação de padrões e fatores determinantes para o não pagamento de débitos.
Nesse sentido, a principal hipótese investigada é se variáveis como altos níveis de endividamento, baixa renda e histórico negativo possuem correlação direta e estatisticamente relevante com o risco de calote por parte de usuários bancários. Assim, essa investigação busca gerar, como produto, a automatização da classificação de risco na concessão de crédito com base nos perfis dos clientes, proporcionando insights estratégicos para decisões mais assertivas e seguras por parte de agências do setor financeiro.

As principais features (variáveis) a serem exploradas são:
- Dados socioeconômicos: Análise de como fatores como experiência profissional, escolaridade, renda anual, taxa de endividamento e a relação crédito/dívida impactam a probabilidade de calote.
- Dados demográficos: Mensuração de como a geolocalização dos clientes se relaciona com o histórico de inadimplência e sua relevância para a classificação de riscos futuros.
- Dados históricos: Verificação da importância do comportamento financeiro passado de cada cliente como preditor plausível de suas ações futuras.

### 📊 Dataset Escolhido
* **Origem:** [Credit Risk Analysis for extending Bank Loans](https://www.kaggle.com/datasets/atulmittal199174/credit-risk-analysis-for-extending-bank-loans)
* **Tamanho:** 1.150 instâncias / 42,07kB
* **Quantidade de Features:** 9 variáveis, incluindo dados socioeconômicos, demográficos e histórico financeiro.

## 👥 Membros da Equipe e Papéis
* **[Anny Caroline Almida Marcelino](https://github.com/AnnyACAM):** Cientista de Dados/ML (Modelagem e avaliação de algoritmos).
* **[Carolina Penido Barcellos](https://github.com/carolinabarcellos):** Analista de Dados (Visualização e interpretação de resultados).
* **[Gabrielly Xavier dos Santos](https://github.com/gabyxsantos):** Engenheira de Dados / Cientista de Dados (Análise exploratória e pré-processamento).
* **[Matheus Soares dos Santos de Freitas](https://github.com/Doctor-Math):** Engenheiro de Dados / Cientista de Dados (Tratamento, organização e preparação do dataset).

## 🛠️ Pilha de Tecnologias

### 💻 Ambiente e Hardware
- **Plataforma:** Google Colab
- **Aceleração de Hardware:** CPU e GPU (NVIDIA T4) para processamento paralelo, conforme necessidade.
- **Memória RAM:** Instância padrão de ~12GB.

### 🐍 Linguagem e Dependências
- **Linguagem:** Python 3.10.12

| Ferramenta | Versão | Função Principal |
| :--- | :--- | :--- |
| **Pandas** | 2.2.2 | Manipulação, limpeza e análise de dados estruturados. |
| **NumPy** | 2.0.2 | Computação numérica e operações com arrays multidimensionais. |
| **Scikit-learn** | 1.6.1 | Modelagem preditiva, pré-processamento e métricas de avaliação. |
| **Matplotlib** | 3.10.0 | Geração de gráficos e customização de visualizações base. |
| **Seaborn** | 0.13.2 | Interface de alto nível para gráficos estatísticos informativos. |
