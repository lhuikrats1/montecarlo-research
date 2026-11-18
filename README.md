# Monte Carlo Portfolio Research

Research-only Python scripts for stock screening, portfolio analysis, scenario
charts, and annual walk-forward backtesting.

## Setup

Use Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp api_keys.env.example api_keys.env
```

Add your Financial Modeling Prep key to `api_keys.env`:

```text
FMP_API_KEY=your_key_here
```

The key file is ignored by Git. Never commit or share your real API keys. Gemini
analysis is optional; add `GEMINI_API_KEY` to `api_keys.env` if you want to use
it.

## Run

Run the regular research workflow:

```bash
python montecarlo.py
```

Run the same analysis with the annual walk-forward backtest:

```bash
python backtest.py
```

Each run saves charts, CSVs, an Excel workbook, and an HTML dashboard locally.
Generated results and virtual environments are excluded from this repository.

## Important

This is a research tool, not investment advice or a guarantee of future
performance. Backtests and Monte Carlo simulations depend on historical data,
model assumptions, transaction costs, and the stock universe supplied to the
scripts. Historical tests using a current stock list may have survivorship bias.
