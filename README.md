# Football Analytics Terminal (BETMAN four-category engine)

## Install (one time)
1. Install Python 3.10+ from python.org (tick "Add Python to PATH").
2. Open a terminal in this folder and run:  `pip install -r requirements.txt`
   (If xgboost fails to install, delete that line: the app falls back to scikit-learn automatically.)

## Run
- Check everything works:   `python tests/test_engine.py`   (10 lines starting with PASS)
- Command-line prediction:  `python cli.py --backtest`
- Dashboard:                `streamlit run app.py`   (opens in your browser)

## Where to put API keys
Copy `.env.example` to `.env`, open it, paste your keys after the `=`:
    API_FOOTBALL_KEY=...   (api-football.com)
    ODDS_API_KEY=...       (the-odds-api.com)
Restart the app. The "Data Sources & API Settings" page shows whether each key is set (never the key itself).
No key needed for football-data.co.uk (free CSV) or CSV upload.

## First real-data steps
Dashboard > Data Sources > "Download & save" (e.g. league E0, season 2324), repeat for several seasons
(aim for 1,000+ matches), reload, then run Historical Backtesting.

## Files
engine/score_model.py  Poisson + Dixon-Coles matrix | engine/betman.py  joint 4-category model
engine/dixon_coles.py  fitting | engine/ml.py  ML features/model | engine/backtest.py  walk-forward validation
engine/odds.py  margin removal, EV | data/providers.py  API/CSV integrations | data/db.py  SQLite
services.py  stats helpers | app.py  dashboard | cli.py  command line

## Known limits
- Injuries/lineups, xG/shots/possession feeds, and league-strength features are not implemented yet.
- ML + ensemble apply to the 1X2 result only; BETMAN/exact-score probabilities come from the Dixon-Coles joint model.
- Halves are assumed independent given team rates.
- Nothing is validated until you backtest on 1,000+ real matches. Probabilities are estimates, not guarantees.
