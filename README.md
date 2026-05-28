# Otimizador de Portfólio de Ativos

App Streamlit para rebalanceamento inteligente de portfólios de investimento, com **alertas** de desvio e cálculo de **compras corretivas** (nunca sugere vendas).

## Funcionalidades

- Define o portfólio com **Tickers**, **Percentagens Alvo** (soma = 100%) e **Quantidades Detidas**
- Obtém preços atuais via [yfinance](https://github.com/ranaroussi/yfinance) (Yahoo Finance)
- Recebe um valor em **euros** a investir
- Calcula a distribuição que aproxima a alocação alvo **sem vender** (apenas compras)
- Mostra alertas de desvio configuráveis
- Compara a alocação atual, alvo e final pós-investimento

## Como correr

```powershell
# Criar virtualenv (recomendado)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Instalar dependências
pip install -r requirements.txt

# Lançar o app
streamlit run app.py
```

Abre automaticamente em `http://localhost:8501`.

## Algoritmo de Rebalanceamento (sem vendas)

Dado:

- $C_i$ — valor atual de cada ativo
- $T_i$ — peso alvo (em %, soma 100)
- $M$ — montante a investir

Pretende-se encontrar $X_i \geq 0$ (compras) tal que, após o investimento, a alocação se aproxime de $T_i$, com $\sum X_i = M$.

**Solução iterativa:**

1. Calcular o valor total pós-investimento $V = \sum C_i + M$.
2. Para cada ativo, ideal $V_i^* = T_i \cdot V$. Se $V_i^* < C_i$, o ativo está sobre-alocado — **fixa-se** ($X_i = 0$).
3. Renormalizar os pesos alvo dos ativos restantes e redistribuir o capital remanescente.
4. Repetir até nenhum ativo novo ficar sobre-alocado.

Garante-se assim que todas as compras são $\geq 0$ e o total investido é exatamente $M$.

## Estrutura

```
.
├── app.py            # App Streamlit + algoritmo
├── requirements.txt  # Dependências
├── .gitignore
└── README.md
```

## Autor

João Fialho — [@joaocruzfialho](https://github.com/joaocruzfialho)
