"""SQLite storage for matches."""
import sqlite3
import pandas as pd
from config import DB_PATH

COLS = ["date", "league", "home", "away", "hg", "ag", "ht_hg", "ht_ag", "oh", "od", "oa", "source", "is_demo"]


def _conn(path=None):
    c = sqlite3.connect(str(path or DB_PATH))
    c.execute("""CREATE TABLE IF NOT EXISTS matches(
        date TEXT, league TEXT, home TEXT, away TEXT, hg INT, ag INT, ht_hg INT, ht_ag INT,
        oh REAL, od REAL, oa REAL, source TEXT, is_demo INT,
        PRIMARY KEY(date, home, away))""")
    return c


def save_matches(df, path=None):
    d = df.copy()
    for c in COLS:
        if c not in d: d[c] = None
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    d["is_demo"] = d["is_demo"].fillna(0).astype(int)
    with _conn(path) as c:
        c.executemany(f"INSERT OR REPLACE INTO matches VALUES({','.join('?'*len(COLS))})",
                      [tuple(None if pd.isna(v) else v for v in r) for r in d[COLS].itertuples(index=False)])
    return len(d)


def load_matches(path=None, include_demo=True):
    with _conn(path) as c:
        df = pd.read_sql("SELECT * FROM matches ORDER BY date", c)
    if not include_demo: df = df[df.is_demo == 0]
    df["date"] = pd.to_datetime(df["date"])
    return df.reset_index(drop=True)
