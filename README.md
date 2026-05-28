# Otimizador de Portfólio de Ativos

App Streamlit para rebalanceamento inteligente de portfólios de investimento, com **alertas**, **histórico**, **comparação de cenários** e **análise de risco**.

## Funcionalidades

- 💼 **Rebalanceamento sem vendas** — algoritmo iterativo que distribui o investimento garantindo apenas compras corretivas
- 💱 **Conversão automática para EUR** via yfinance (USD, GBP, HKD, ...)
- 🍩 **Donuts comparativos** — alocação atual vs. alvo vs. final
- 🩺 **Score de saúde** + alerta de concentração + sugestão de urgência
- 💾 **Persistência** — `data/portfolio.json` (auto-save) + import/export JSON
- 📈 **Histórico de snapshots** — gravados em `data/history.json` em cada *aplicar*; gráficos de evolução do valor e alocação
- 🎯 **Comparação de cenários** — calcular rebalanceamento para vários valores em € lado a lado
- ⚖️ **Análise de risco** — volatilidade anualizada, matriz de correlação, métricas de portfólio (retorno, vol, Sharpe)

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

Abre em `http://localhost:8501`.

## Estrutura

```
.
├── app.py            # App Streamlit (UI + algoritmo + análises)
├── requirements.txt  # streamlit, yfinance, pandas, numpy, plotly
├── data/             # portfolio.json + history.json (gitignored)
├── .gitignore
└── README.md
```

## Algoritmo de Rebalanceamento (sem vendas)

Dado $C_i$ (valor atual), $T_i$ (peso alvo %), $M$ (montante a investir), encontra-se $X_i \geq 0$ (compras) tal que a alocação final aproxime $T_i$ com $\sum X_i = M$.

Solução iterativa:

1. $V = \sum C_i + M$. Ideal $V_i^* = T_i \cdot V$.
2. Ativos com $V_i^* < C_i$ → fixa-se $X_i = 0$ (sobre-alocados).
3. Renormaliza pesos alvo dos restantes e redistribui o capital remanescente.
4. Repete até convergir. Garante $X_i \geq 0$ e $\sum X_i = M$ exato.

## Análise de Risco

- Preços históricos via `yf.download(...)` (cache de 30 min)
- Retornos diários e estatísticas anualizadas (252 dias úteis)
- Matriz de correlação entre retornos diários
- Métricas de portfólio com pesos = alvo (assume EUR; ignora risco cambial)

> **Nota:** Para ativos não-EUR, a volatilidade calculada é em moeda nativa; o risco cambial adiciona-se à exposição real do investidor europeu.

## Autor

João Fialho — [@joaocruzfialho](https://github.com/joaocruzfialho)
