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
    """Plain-language warnings about the loaded data."""
    w = []
    if df.empty: return ["No data loaded."]
    if df.get("is_demo", pd.Series([0])).astype(bool).any(): w.append("Contains DEMO (simulated) matches. Not real results.")
    if len(df) < 1000: w.append(f"Only {len(df)} matches: backtesting needs 1,000+ for a meaningful validation.")
    age = (pd.Timestamp.today() - pd.to_datetime(df.date).max()).days
    if age > 30: w.append(f"Newest match is {age} days old: data may be outdated.")
    if "ht_hg" not in df or df.ht_hg.isna().mean() > 0.2: w.append("Half-time scores are largely missing: first-half shares use an assumed default (45%).")
    if "oh" not in df or df.oh.isna().all(): w.append("No bookmaker odds in data: odds comparison unavailable.")
    return w


def explain(home, away, lh, la, rho, share, hs, as_):
    """Plain-language list of the factors behind a prediction."""
    fav = home if lh > la else away
    lines = [f"Expected goals: {home} {lh:.2f}, {away} {la:.2f}. {fav} is the statistical favourite on goal rates.",
             f"Roughly {share*100:.0f}% of goals are modelled in the first half (observed or assumed default).",
             f"Low-score correction (Dixon-Coles rho = {rho:+.3f}) adjusts 0-0, 1-0, 0-1 and 1-1 probabilities."]
    if hs: lines.append(f"{home}: {hs['goals_for']:.2f} scored / {hs['goals_against']:.2f} conceded per match in the data ({hs['matches']} matches).")
    if as_: lines.append(f"{away}: {as_['goals_for']:.2f} scored / {as_['goals_against']:.2f} conceded per match in the data ({as_['matches']} matches).")
    lines.append("Injuries, suspensions and lineups are NOT in this model yet; check them yourself before relying on any number.")
    return lines
