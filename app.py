"""Run:  streamlit run app.py"""
import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
import config, services as sv
from data import db
from data.demo_data import make_demo
from data.providers import (ApiFootball, OddsApi, ProviderError, fetch_football_data, load_csv)
from engine.backtest import run_backtest
from engine.betman import CATS, betman_joint, expected_score, marginals, top_combinations, uncertainty
from engine.dixon_coles import DixonColes, first_half_share
from engine.odds import compare, overround
from engine.score_model import btts, over_under, result_probs, score_matrix

st.set_page_config(page_title="Football Analytics Terminal", layout="wide", page_icon="⚽")
st.markdown("<style>.stMetric{background:#161b22;padding:10px;border-radius:8px}</style>", unsafe_allow_html=True)
DISCLAIMER = "Model estimates only: not certainties, not advice, no guarantee of profit. Never stake money you cannot afford to lose."

# ---------- data ----------
@st.cache_data(show_spinner=False)
def get_data(version):
    real = db.load_matches(include_demo=False)
    return (real, False) if len(real) else (make_demo(seasons=4), True)

if "ver" not in st.session_state: st.session_state.ver = 0
df, is_demo = get_data(st.session_state.ver)

@st.cache_resource(show_spinner=False)
def fit(_df, key): return DixonColes().fit(_df)

page = st.sidebar.radio("Navigate", ["Dashboard", "Upcoming Matches", "Match Analysis", "BETMAN Prediction", "Exact Score",
                                     "Odds Intelligence", "Team Statistics", "Historical Backtesting", "Model Performance",
                                     "Data Sources & API Settings"])
if is_demo: st.sidebar.warning("DEMO DATA (simulated). Load real data in 'Data Sources'.")
try:
    dc = fit(df, (len(df), str(df.date.max())))
except ValueError as e:
    dc = None; st.error(str(e))
share = first_half_share(df) or 0.45

def pick_match(key=""):
    c1, c2 = st.columns(2)
    h = c1.selectbox("Home team", dc.teams, 0, key="h" + key)
    a = c2.selectbox("Away team", [t for t in dc.teams if t != h], 0, key="a" + key)
    lh, la = dc.rates(h, a); return h, a, lh, la

def need_model():
    if dc is None: st.stop()

# ---------- pages ----------
if page == "Dashboard":
    st.title("Dashboard")
    for w in sv.data_quality(df): st.warning(w)
    c = st.columns(4); c[0].metric("Matches loaded", len(df)); c[1].metric("Teams", df.home.nunique())
    c[2].metric("Avg goals/match", f"{(df.hg + df.ag).mean():.2f}"); c[3].metric("Home win %", f"{(df.hg > df.ag).mean()*100:.0f}")
    st.subheader("League table (from loaded data)"); st.dataframe(sv.league_table(df), use_container_width=True)
    st.caption(DISCLAIMER)

elif page == "Upcoming Matches":
    st.title("Upcoming Matches")
    if not config.get_key("API_FOOTBALL_KEY"):
        st.info("No API-Football key set, so no live fixtures. Add it to the .env file (see 'Data Sources & API Settings'), or upload fixtures below.")
    else:
        lid = st.number_input("League id (39 = Premier League)", 1, 2000, 39); season = st.number_input("Season", 2015, 2030, 2025)
        if st.button("Fetch upcoming fixtures"):
            try: st.session_state.fix = ApiFootball().upcoming(int(lid), int(season))
            except ProviderError as e: st.error(str(e))
    up = st.file_uploader("Or upload fixtures CSV (home,away[,date])", type="csv")
    fx = pd.read_csv(up) if up else st.session_state.get("fix")
    if fx is not None and len(fx) and dc:
        rows = []
        for r in fx.itertuples():
            if r.home in dc.idx and r.away in dc.idx:
                lh, la = dc.rates(r.home, r.away); p = result_probs(score_matrix(lh, la, dc.rho))
                rows.append(dict(home=r.home, away=r.away, xG_home=round(lh, 2), xG_away=round(la, 2), **{k + " %": round(v * 100, 1) for k, v in p.items()}))
            else: rows.append(dict(home=r.home, away=r.away, note="team not in historical data: no prediction"))
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

elif page == "Match Analysis":
    st.title("Match Analysis"); need_model(); h, a, lh, la = pick_match()
    J = betman_joint(lh, la, share, dc.rho); m = score_matrix(lh, la, dc.rho)
    st.subheader("Why this prediction"); [st.write("• " + x) for x in sv.explain(h, a, lh, la, dc.rho, share, sv.team_summary(df, h, 10), sv.team_summary(df, a, 10))]
    c1, c2 = st.columns(2)
    for col, t in ((c1, h), (c2, a)):
        col.markdown(f"**{t}: last 5**"); col.dataframe(sv.last_results(df, t, 5), hide_index=True)
        col.markdown("Last 10 / home / away"); col.dataframe(pd.DataFrame({"last10": sv.team_summary(df, t, 10), "home": sv.team_summary(df, t, venue="home"), "away": sv.team_summary(df, t, venue="away")}).round(2))
    st.subheader("Head to head"); st.dataframe(sv.head_to_head(df, h, a), hide_index=True)
    p = result_probs(m); st.plotly_chart(px.bar(x=["Home", "Draw", "Away"], y=[v * 100 for v in p.values()], template="plotly_dark", labels=dict(x="", y="%")), use_container_width=True)
    st.subheader("Top 10 BETMAN combinations"); st.dataframe(pd.DataFrame([(c, round(p_ * 100, 2)) for p_, c in top_combinations(J)], columns=["1H h-a-FT h-a", "%"]), hide_index=True)
    st.caption(DISCLAIMER)

elif page == "BETMAN Prediction":
    st.title("BETMAN Four-Category Prediction"); need_model(); h, a, lh, la = pick_match("b")
    J = betman_joint(lh, la, share, dc.rho); M = marginals(J)
    st.caption(f"Joint model: 1H and 2H scoring modelled separately; FT = 1H + 2H, so FT can never be below HT. 1H goal share {share*100:.0f}%.")
    lab = {"ht_home": f"1H · {h}", "ht_away": f"1H · {a}", "ft_home": f"FT · {h}", "ft_away": f"FT · {a}"}
    cols = st.columns(4)
    for c, (k, v) in zip(cols, M.items()):
        b = int(v.argmax()); c.markdown(f"**{lab[k]}**")
        c.dataframe(pd.DataFrame({"Goals": CATS, "Prob %": (v * 100).round(1)}), hide_index=True)
        c.metric("Most likely", CATS[b], f"{v[b]*100:.1f}%"); c.caption(f"Uncertainty (spread) {uncertainty(v):.2f}  (0 = concentrated, 1 = spread out)")
    es = expected_score(J); st.write(f"Expected goals: 1H {es['ht_home']:.2f}-{es['ht_away']:.2f} · FT {es['ft_home']:.2f}-{es['ft_away']:.2f}")
    top = top_combinations(J, 10); st.subheader("Top 10 joint combinations")
    st.dataframe(pd.DataFrame([(c, round(p * 100, 2)) for p, c in top], columns=["1H home-away-FT home-away", "Joint %"]), hide_index=True)
    st.plotly_chart(px.bar(x=[c for _, c in top], y=[p * 100 for p, _ in top], template="plotly_dark", labels=dict(x="combination", y="%")), use_container_width=True)
    pick = st.text_input("Check your slip (e.g. 0-0-0-1)", "0-0-0-1")
    try:
        idx = tuple(CATS.index(x) for x in pick.split("-")); assert len(idx) == 4
        st.metric("Probability of your combination", f"{J[idx]*100:.2f}%", "impossible: FT below HT" if J[idx] == 0 else None)
    except Exception: st.caption("Use format 1H-home, 1H-away, FT-home, FT-away with values 0,1,2,3,4,5+")
    st.caption("Spread measures how uncertain the model's distribution is. It is NOT a validated accuracy. " + DISCLAIMER)

elif page == "Exact Score":
    st.title("Exact Score"); need_model(); h, a, lh, la = pick_match("e"); m = score_matrix(lh, la, dc.rho)
    f = share; m1 = score_matrix(lh * f, la * f, dc.rho)
    for title, mm in (("Full-time", m), ("Half-time", m1)):
        st.subheader(title); n = 6; fig = go.Figure(go.Heatmap(z=mm[:n, :n] * 100, text=np.round(mm[:n, :n] * 100, 1), texttemplate="%{text}", colorscale="Viridis"))
        fig.update_layout(template="plotly_dark", xaxis_title=f"{a} goals", yaxis_title=f"{h} goals"); st.plotly_chart(fig, use_container_width=True)
        flat = sorted(((mm[i, j], f"{i}-{j}") for i in range(n) for j in range(n)), reverse=True)[:10]
        st.dataframe(pd.DataFrame([(s, round(p * 100, 2)) for p, s in flat], columns=["Score", "%"]), hide_index=True)
    c = st.columns(5)
    for col, line in zip(c, (0.5, 1.5, 2.5, 3.5, 4.5)): col.metric(f"Over {line}", f"{over_under(m, line)['over']*100:.0f}%")
    st.metric("BTTS", f"{btts(m)*100:.0f}%")

elif page == "Odds Intelligence":
    st.title("Odds Intelligence"); need_model(); h, a, lh, la = pick_match("o")
    st.write("Enter decimal odds (e.g. from your bookmaker's site, or fetched below).")
    c = st.columns(3); oh = c[0].number_input("Home", 1.01, 100.0, 2.30); od = c[1].number_input("Draw", 1.01, 100.0, 3.30); oa = c[2].number_input("Away", 1.01, 100.0, 3.10)
    p = list(result_probs(score_matrix(lh, la, dc.rho)).values()); t = pd.DataFrame(compare(p, oh, od, oa))
    st.dataframe(t.assign(**{c_: (t[c_] * 100).round(1) for c_ in ("model", "market", "diff")}).round(3), hide_index=True)
    st.write(f"Bookmaker margin (overround): {overround(oh, od, oa)*100:.1f}%")
    st.warning("EV assumes the model is exactly right, which it is not. A positive EV does NOT mean profit. This tool places no bets and connects to no betting accounts.")
    if st.button("Fetch live odds (The Odds API)"):
        try: st.dataframe(OddsApi().h2h_odds())
        except ProviderError as e: st.error(str(e))

elif page == "Team Statistics":
    st.title("Team Statistics"); need_model(); t = st.selectbox("Team", dc.teams)
    st.dataframe(pd.DataFrame({"all": sv.team_summary(df, t), "last 5": sv.team_summary(df, t, 5), "last 10": sv.team_summary(df, t, 10),
                               "home": sv.team_summary(df, t, venue="home"), "away": sv.team_summary(df, t, venue="away")}).round(2))
    i = dc.idx[t]; st.write(f"Model strengths: attack {dc.att[i]:+.2f}, defence {dc.dfn[i]:+.2f} (lower defence = harder to score against). Home advantage {dc.home:+.2f}.")

elif page == "Historical Backtesting":
    st.title("Historical Backtesting")
    st.write("Walk-forward test: the model is trained only on past matches, then predicts the next block, repeatedly. No future data is used.")
    c = st.columns(3); mt = c[0].number_input("Min training matches", 100, 5000, 300); stp = c[1].number_input("Refit every N matches", 10, 500, 50); ml_on = c[2].checkbox("Include ML + ensemble", True)
    if st.button("Run backtest"):
        bar = st.progress(0.0); st.session_state.bt = run_backtest(df, int(mt), int(stp), ml_on, bar.progress)
    r = st.session_state.get("bt")
    if r: st.success("Done. See 'Model Performance'.") if r.get("n_evaluated") else st.error(r["message"])

elif page == "Model Performance":
    st.title("Model Performance"); r = st.session_state.get("bt")
    if not r or not r.get("n_evaluated"): st.info("No backtest results yet. Run one in 'Historical Backtesting'. Until then the model is NOT validated and no accuracy is shown.")
    else:
        (st.success if r["validated"] else st.warning)(r["message"])
        if is_demo: st.error("These results are on SIMULATED demo data and say nothing about real football.")
        st.subheader("Result (1X2) quality: lower Brier / log loss / calibration error is better")
        st.dataframe(pd.DataFrame(r["1x2"]).T.round(4)); st.caption("Naive = always predict past home/draw/away frequencies. A useful model must beat it.")
        if "mean_ml_weight" in r: st.caption(f"Average ML weight in ensemble (chosen from past out-of-sample results): {r['mean_ml_weight']:.2f}")
        c = st.columns(3)
        c[0].write("Exact score"); c[0].json({k: round(v, 4) for k, v in r["exact_score"].items()})
        if "half_time_1x2" in r: c[1].write("Half-time 1X2"); c[1].json({k: round(v, 4) for k, v in r["half_time_1x2"].items()})
        if "betman_joint" in r: c[2].write("BETMAN 4-category joint"); c[2].json({k: round(v, 4) for k, v in r["betman_joint"].items()})
        rel = r["reliability"]; fig = px.scatter(rel, x="predicted", y="observed", size="count", template="plotly_dark", title="Calibration (closer to the diagonal is better)")
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash")); st.plotly_chart(fig, use_container_width=True)

else:
    st.title("Data Sources & API Settings")
    st.subheader("1. API keys: where to put them")
    st.code("1. In the project folder, copy  .env.example  to  .env\n2. Open .env in Notepad / any text editor\n3. Replace PASTE_YOUR_..._HERE with your key (no quotes, no spaces)\n4. Save, then restart the app", language="text")
    for name, cls in (("API_FOOTBALL_KEY", ApiFootball), ("ODDS_API_KEY", OddsApi)):
        ok = bool(config.get_key(name)); st.write(f"**{name}**: {'✅ set' if ok else '❌ not set'}")
    if st.button("Test API-Football connection"):
        try: ApiFootball().status(); st.success("Connected.")
        except ProviderError as e: st.error(str(e))
    st.subheader("2. Football-Data.co.uk (free, no key)")
    c = st.columns(3); lg = c[0].text_input("League code", "E0"); ss = c[1].text_input("Season (2324 = 2023/24)", "2324")
    if c[2].button("Download & save"):
        try: n = db.save_matches(fetch_football_data(ss, lg)); st.session_state.ver += 1; get_data.clear(); st.success(f"Saved {n} matches. Reload the page.")
        except ProviderError as e: st.error(str(e))
    st.subheader("3. Upload CSV")
    up = st.file_uploader("CSV (football-data.co.uk format, or date,home,away,hg,ag[,ht_hg,ht_ag])", type="csv")
    if up and st.button("Save uploaded file"):
        try: n = db.save_matches(load_csv(up)); st.session_state.ver += 1; get_data.clear(); st.success(f"Saved {n} matches. Reload the page.")
        except ProviderError as e: st.error(str(e))
    st.subheader("4. API-Football history")
    c = st.columns(3); lid = c[0].number_input("League id", 1, 2000, 39); se = c[1].number_input("Season", 2015, 2030, 2024)
    if c[2].button("Import from API-Football"):
        try: n = db.save_matches(ApiFootball().finished_matches(int(lid), int(se))); get_data.clear(); st.success(f"Saved {n} matches.")
        except ProviderError as e: st.error(str(e))
    st.subheader("Data quality"); [st.warning(w) for w in sv.data_quality(df)] or st.success("No issues found.")
