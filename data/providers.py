"""Data providers. Nothing here invents data: if a source is unavailable we raise a clear error."""
import io
import pandas as pd
import requests
from config import get_key


class ProviderError(Exception):
    pass


def _get(url, **kw):
    try:
        r = requests.get(url, timeout=20, **kw)
    except requests.RequestException as e:
        raise ProviderError(f"Network problem: {e}")
    if r.status_code == 429: raise ProviderError("Rate limit reached (HTTP 429). Wait and retry.")
    if r.status_code in (401, 403): raise ProviderError(f"Key rejected (HTTP {r.status_code}). Check your key.")
    if r.status_code != 200: raise ProviderError(f"HTTP {r.status_code} from {url.split('?')[0]}")
    return r


# ---------- Football-Data.co.uk (free CSV, no key) ----------
def normalize_football_data(df, league=None):
    """Convert football-data.co.uk columns to our schema. Missing columns stay missing."""
    need = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    if not need.issubset(df.columns):
        raise ProviderError(f"CSV is missing columns: {sorted(need - set(df.columns))}")
    d = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    out = pd.DataFrame({
        "date": pd.to_datetime(d["Date"], dayfirst=True, errors="coerce"),
        "league": league or d.get("Div", pd.Series([None] * len(d), index=d.index)),
        "home": d["HomeTeam"], "away": d["AwayTeam"],
        "hg": d["FTHG"].astype(int), "ag": d["FTAG"].astype(int),
        "ht_hg": d["HTHG"] if "HTHG" in d else None, "ht_ag": d["HTAG"] if "HTAG" in d else None,
    })
    for prefix in ("B365", "Avg", "PS"):                      # first odds set that exists
        if {prefix + "H", prefix + "D", prefix + "A"}.issubset(d.columns):
            out["oh"], out["od"], out["oa"] = d[prefix + "H"], d[prefix + "D"], d[prefix + "A"]
            break
    out["source"] = "football-data.co.uk"; out["is_demo"] = 0
    return out.dropna(subset=["date"]).reset_index(drop=True)


def fetch_football_data(season="2324", league_code="E0"):
    """season '2324' = 2023/24; league E0 = Premier League. Public CSV, no key."""
    r = _get(f"https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv")
    return normalize_football_data(pd.read_csv(io.BytesIO(r.content), encoding="latin-1"), league_code)


def load_csv(file_or_path):
    """Our own schema (date,home,away,hg,ag,...) OR a football-data.co.uk file."""
    df = pd.read_csv(file_or_path, encoding="latin-1")
    if {"HomeTeam", "FTHG"}.issubset(df.columns): return normalize_football_data(df)
    miss = {"date", "home", "away", "hg", "ag"} - set(df.columns)
    if miss: raise ProviderError(f"CSV needs columns date,home,away,hg,ag (missing {sorted(miss)})")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["source"] = "csv-upload"; df["is_demo"] = 0
    return df.dropna(subset=["date"]).reset_index(drop=True)


# ---------- API-Football (needs API_FOOTBALL_KEY in .env) ----------
class ApiFootball:
    BASE = "https://v3.football.api-sports.io"

    def __init__(self):
        self.key = get_key("API_FOOTBALL_KEY")

    def _call(self, path, params):
        if not self.key:
            raise ProviderError("API_FOOTBALL_KEY not set. Put it in the .env file.")
        r = _get(f"{self.BASE}/{path}", headers={"x-apisports-key": self.key}, params=params)
        j = r.json()
        if j.get("errors"): raise ProviderError(f"API-Football error: {j['errors']}")
        return j.get("response", [])

    def status(self):
        return self._call("status", {})

    def finished_matches(self, league_id, season):
        rows = []
        for f in self._call("fixtures", {"league": league_id, "season": season, "status": "FT"}):
            rows.append(dict(date=f["fixture"]["date"][:10], league=str(league_id),
                             home=f["teams"]["home"]["name"], away=f["teams"]["away"]["name"],
                             hg=f["goals"]["home"], ag=f["goals"]["away"],
                             ht_hg=f["score"]["halftime"]["home"], ht_ag=f["score"]["halftime"]["away"],
                             source="api-football", is_demo=0))
        return pd.DataFrame(rows)

    def upcoming(self, league_id, season, n=20):
        return pd.DataFrame([dict(date=f["fixture"]["date"][:10], home=f["teams"]["home"]["name"],
                                  away=f["teams"]["away"]["name"], fixture_id=f["fixture"]["id"])
                             for f in self._call("fixtures", {"league": league_id, "season": season, "next": n})])


# ---------- The Odds API (needs ODDS_API_KEY in .env) ----------
class OddsApi:
    BASE = "https://api.the-odds-api.com/v4"

    def __init__(self):
        self.key = get_key("ODDS_API_KEY")

    def h2h_odds(self, sport="soccer_epl", regions="eu"):
        if not self.key: raise ProviderError("ODDS_API_KEY not set. Put it in the .env file.")
        r = _get(f"{self.BASE}/sports/{sport}/odds", params=dict(apiKey=self.key, regions=regions, markets="h2h", oddsFormat="decimal"))
        rows = []
        for ev in r.json():
            for bk in ev.get("bookmakers", []):
                o = {x["name"]: x["price"] for x in bk["markets"][0]["outcomes"]} if bk.get("markets") else {}
                if ev["home_team"] in o and ev["away_team"] in o and "Draw" in o:
                    rows.append(dict(date=ev["commence_time"][:10], home=ev["home_team"], away=ev["away_team"],
                                     bookmaker=bk["title"], oh=o[ev["home_team"]], od=o["Draw"], oa=o[ev["away_team"]]))
        return pd.DataFrame(rows)
