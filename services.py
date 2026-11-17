"""Plain functions the dashboard uses (kept out of the UI so they can be tested)."""
import numpy as np
import pandas as pd


def team_matches(df, team, n=None, venue=None):
    m = df[(df.home == team) | (df.away == team)]
    if venue == "home": m = m[m.home == team]
    if venue == "away": m = m[m.away == team]
    m = m.sort_values("date")
    return m.tail(n) if n else m


def team_summary(df, team, n=None, venue=None):
    m = team_matches(df, team, n, venue)
    if m.empty: return {}
    ish = m.home == team
    gf = np.where(ish, m.hg, m.ag); ga = np.where(ish, m.ag, m.hg)
    d = dict(matches=len(m), wins=int((gf > ga).sum()), draws=int((gf == ga).sum()), losses=int((gf < ga).sum()),
             goals_for=float(gf.mean()), goals_against=float(ga.mean()), clean_sheets=float((ga == 0).mean()),
             btts=float(((gf > 0) & (ga > 0)).mean()), over25=float(((gf + ga) > 2.5).mean()))
    if "ht_hg" in m and m.ht_hg.notna().all():
        h1f = np.where(ish, m.ht_hg, m.ht_ag); h1a = np.where(ish, m.ht_ag, m.ht_hg)
        d.update(ht_goals_for=float(h1f.mean()), ht_goals_against=float(h1a.mean()))
    return d


def last_results(df, team, n=5):
    m = team_matches(df, team, n).copy()
    ish = m.home == team
    m["opponent"] = np.where(ish, m.away, m.home); m["venue"] = np.where(ish, "H", "A")
    m["score"] = m.hg.astype(str) + "-" + m.ag.astype(str)
    gf = np.where(ish, m.hg, m.ag); ga = np.where(ish, m.ag, m.hg)
    m["result"] = np.where(gf > ga, "W", np.where(gf == ga, "D", "L"))
    return m[["date", "opponent", "venue", "score", "result"]]


def head_to_head(df, a, b, n=10):
    m = df[((df.home == a) & (df.away == b)) | ((df.home == b) & (df.away == a))].sort_values("date").tail(n)
    return m[["date", "home", "away", "hg", "ag"]]


def league_table(df):
    t = {}
    for r in df.itertuples():
        for team, gf, ga in ((r.home, r.hg, r.ag), (r.away, r.ag, r.hg)):
            x = t.setdefault(team, dict(P=0, W=0, D=0, L=0, GF=0, GA=0, Pts=0))
            x["P"] += 1; x["GF"] += gf; x["GA"] += ga
            k = "W" if gf > ga else "D" if gf == ga else "L"; x[k] += 1; x["Pts"] += 3 if k == "W" else 1 if k == "D" else 0
    out = pd.DataFrame(t).T; out["GD"] = out.GF - out.GA
    return out.sort_values(["Pts", "GD", "GF"], ascending=False)


def data_quality(df):
    """Warnings as (code, params); the UI translates them."""
    w = []
    if df.empty: return [("dq_none", {})]
    if df.get("is_demo", pd.Series([0])).astype(bool).any(): w.append(("dq_demo", {}))
    if len(df) < 1000: w.append(("dq_small", {"n": len(df)}))
    age = (pd.Timestamp.today() - pd.to_datetime(df.date).max()).days
    if age > 30: w.append(("dq_old", {"days": age}))
    if "ht_hg" not in df or df.ht_hg.isna().mean() > 0.2: w.append(("dq_ht", {}))
    if "oh" not in df or df.oh.isna().all(): w.append(("dq_odds", {}))
    return w
