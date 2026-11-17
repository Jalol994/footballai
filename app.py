"""Run:  streamlit run app.py"""
import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
import config, services as sv, ui
from data import db
from data.demo_data import make_demo
from data.providers import ApiFootball, OddsApi, ProviderError, fetch_football_data, load_csv
from engine.backtest import run_backtest
from engine.betman import CATS, betman_joint, expected_score, marginals, top_combinations, uncertainty
from engine.dixon_coles import DixonColes, first_half_share
from engine.odds import compare, overround
from engine.score_model import btts, over_under, result_probs, score_matrix
from i18n import LANGS, make_t

st.set_page_config(page_title="Football Analytics", layout="wide", page_icon="⚽")
st.markdown(f"<style>{ui.CSS}</style>", unsafe_allow_html=True)

lang = st.sidebar.radio("🌐", list(LANGS), format_func=lambda c: LANGS[c], horizontal=True, key="lang")
t = make_t(lang)
PAGES = ["dashboard", "upcoming", "analysis", "betman", "exact", "odds", "teams", "backtest", "perf", "sources"]
page = st.sidebar.radio(t("navigate"), PAGES, format_func=lambda k: f"{ui.ICONS[k]}  {t('nav_' + k)}")
md = lambda h: st.markdown(h, unsafe_allow_html=True)
md(ui.hero(f"{ui.ICONS[page]} {t('nav_' + page)}", t("app_sub")))


def style(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=30, b=10))
    return fig


# ---------- data ----------
@st.cache_data(show_spinner=False)
def get_data(version):
    real = db.load_matches(include_demo=False)
    return (real, False) if len(real) else (make_demo(seasons=4), True)

if "ver" not in st.session_state: st.session_state.ver = 0
df, is_demo = get_data(st.session_state.ver)

@st.cache_resource(show_spinner=False)
def fit(_df, key): return DixonColes().fit(_df)

if is_demo: st.sidebar.warning(t("demo_side"))
try:
    dc = fit(df, (len(df), str(df.date.max())))
except ValueError as e:
    dc = None; st.error(str(e))
share = first_half_share(df) or 0.45

TL = {}
if "league" in df:
    for team in pd.unique(pd.concat([df.home, df.away])):
        s = pd.concat([df.loc[df.home == team, "league"], df.loc[df.away == team, "league"]]).dropna()
        if len(s): TL[team] = str(s.mode().iloc[0])
flag = lambda x: ui.team_flag(x, TL.get(x))
label = lambda x: f"{flag(x)} {x}".strip()


def need_model():
    if dc is None: st.stop()


def pick_match(key=""):
    c1, c2 = st.columns(2)
    h = c1.selectbox(t("home_team"), dc.teams, 0, key="h" + key, format_func=label)
    a = c2.selectbox(t("away_team"), [x for x in dc.teams if x != h], 0, key="a" + key, format_func=label)
    lh, la = dc.rates(h, a)
    p = result_probs(score_matrix(lh, la, dc.rho))
    md(ui.match_banner(h, a, flag(h), flag(a), lh, la, p, (t("home_win"), t("draw"), t("away_win"))))
    return h, a, lh, la


def stats_table(**cols):
    d = pd.DataFrame(cols)
    d.index = [t("stat_" + i) for i in d.index]
    return d.round(2)

# ---------- pages ----------
if page == "dashboard":
    for code, kw in sv.data_quality(df): md(ui.warn(t(code, **kw)))
    c = st.columns(4)
    for col, (lab, val) in zip(c, [(t("matches_loaded"), len(df)), (t("teams"), df.home.nunique()),
                                   (t("avg_goals"), f"{(df.hg + df.ag).mean():.2f}"), (t("home_win_pct"), f"{(df.hg > df.ag).mean() * 100:.0f}")]):
        col.markdown(ui.card(lab, val), unsafe_allow_html=True)
    st.subheader(t("league_table"))
    tbl = sv.league_table(df); tbl.index = [label(x) for x in tbl.index]
    st.dataframe(tbl, use_container_width=True)
    md(ui.disclaimer(t("disclaimer"))); md(ui.disclaimer(t("flag_note")))

elif page == "upcoming":
    if not config.get_key("API_FOOTBALL_KEY"):
        st.info(t("up_nokey"))
    else:
        lid = st.number_input(t("league_id"), 1, 2000, 39); season = st.number_input(t("season"), 2015, 2030, 2025)
        if st.button(t("fetch_fx")):
            try: st.session_state.fix = ApiFootball().upcoming(int(lid), int(season))
            except ProviderError as e: st.error(str(e))
    up = st.file_uploader(t("up_csv"), type="csv")
    fx = pd.read_csv(up) if up else st.session_state.get("fix")
    if fx is not None and len(fx) and dc:
        rows = []
        for r in fx.itertuples():
            if r.home in dc.idx and r.away in dc.idx:
                lh, la = dc.rates(r.home, r.away); p = result_probs(score_matrix(lh, la, dc.rho))
                rows.append({t("col_home"): label(r.home), t("col_away"): label(r.away), "xG 1": round(lh, 2), "xG 2": round(la, 2),
                             t("home_win") + " %": round(p["home"] * 100, 1), t("draw") + " %": round(p["draw"] * 100, 1), t("away_win") + " %": round(p["away"] * 100, 1)})
            else: rows.append({t("col_home"): r.home, t("col_away"): r.away, "": t("no_pred")})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

elif page == "analysis":
    need_model(); h, a, lh, la = pick_match(); J = betman_joint(lh, la, share, dc.rho)
    fav = h if lh > la else a
    st.subheader(t("why"))
    hs, as_ = sv.team_summary(df, h, 10), sv.team_summary(df, a, 10)
    lines = [t("ex_xg", h=h, a=a, lh=lh, la=la, fav=fav), t("ex_share", p=share * 100), t("ex_rho", rho=dc.rho)]
    for tm, s_ in ((h, hs), (a, as_)):
        if s_: lines.append(t("ex_team", t=tm, gf=s_["goals_for"], ga=s_["goals_against"], n=s_["matches"]))
    lines.append(t("ex_missing")); [st.write("• " + x) for x in lines]
    c1, c2 = st.columns(2)
    for col, tm in ((c1, h), (c2, a)):
        col.markdown(f"**{label(tm)}: {t('last5')}**"); col.dataframe(sv.last_results(df, tm, 5), hide_index=True)
        col.markdown(t("last10"))
        col.dataframe(stats_table(**{t("col_l10"): sv.team_summary(df, tm, 10), t("col_homev"): sv.team_summary(df, tm, venue="home"), t("col_awayv"): sv.team_summary(df, tm, venue="away")}))
    st.subheader(t("h2h")); st.dataframe(sv.head_to_head(df, h, a), hide_index=True)
    st.subheader(t("top10_b"))
    st.dataframe(pd.DataFrame([(c, round(p_ * 100, 2)) for p_, c in top_combinations(J)], columns=[t("combo"), t("pct")]), hide_index=True)
    md(ui.disclaimer(t("disclaimer")))

elif page == "betman":
    need_model(); h, a, lh, la = pick_match("b")
    J = betman_joint(lh, la, share, dc.rho); M = marginals(J); top = top_combinations(J, 10)
    best_p, best_c = top[0]; digits = best_c.split("-")
    st.subheader("🎯 " + t("quick_title"))
    keys = ["ht_home", "ht_away", "ft_home", "ft_away"]
    names = [f"{t('bm_ht')} · {h}", f"{t('bm_ht')} · {a}", f"{t('bm_ft')} · {h}", f"{t('bm_ft')} · {a}"]
    md(ui.slip([(n, d, M[k][CATS.index(d)] * 100) for n, d, k in zip(names, digits, keys)]))
    st.caption(t("quick_hint", c=best_c)); st.write(t("one_in", p=best_p * 100, n=round(1 / best_p)))
    md(ui.warn(t("quick_warn")))
    st.markdown("**" + t("alt") + "**")
    st.dataframe(pd.DataFrame([(c, round(p * 100, 2), f"1 / {round(1 / p)}") for p, c in top[1:4]], columns=[t("combo"), t("pct"), "1 / N"]), hide_index=True)
    st.subheader(t("details")); st.caption(t("bm_caption", s=share * 100))
    cols = st.columns(4)
    for col, n, k in zip(cols, names, keys):
        v = M[k]; col.markdown(ui.bars(n, CATS, v), unsafe_allow_html=True)
        col.caption(f"{t('spread')}: {uncertainty(v):.2f} ({t('spread_help')})")
    es = expected_score(J); st.write(f"{t('exp_goals')}: {t('bm_ht')} {es['ht_home']:.2f}-{es['ht_away']:.2f} · {t('bm_ft')} {es['ft_home']:.2f}-{es['ft_away']:.2f}")
    st.subheader(t("top10"))
    st.dataframe(pd.DataFrame([(c, round(p * 100, 2)) for p, c in top], columns=[t("combo"), t("joint")]), hide_index=True)
    st.plotly_chart(style(px.bar(x=[c for _, c in top], y=[p * 100 for p, _ in top], labels=dict(x="", y="%"))), use_container_width=True)
    pick = st.text_input(t("check_slip"), "0-0-0-1")
    try:
        idx = tuple(CATS.index(x) for x in pick.strip().split("-")); assert len(idx) == 4
        st.metric(t("slip_prob"), f"{J[idx] * 100:.2f}%", t("impossible") if J[idx] == 0 else None)
    except Exception: st.caption(t("slip_fmt"))
    st.caption(t("spread_note")); md(ui.disclaimer(t("disclaimer"))); md(ui.disclaimer(t("no_bet")))

elif page == "exact":
    need_model(); h, a, lh, la = pick_match("e"); m = score_matrix(lh, la, dc.rho); m1 = score_matrix(lh * share, la * share, dc.rho)
    for title, mm in ((t("fulltime"), m), (t("halftime"), m1)):
        st.subheader(title); n = 6
        fig = go.Figure(go.Heatmap(z=mm[:n, :n] * 100, text=np.round(mm[:n, :n] * 100, 1), texttemplate="%{text}", colorscale="Viridis"))
        fig.update_layout(xaxis_title=a, yaxis_title=h); st.plotly_chart(style(fig), use_container_width=True)
        flat = sorted(((mm[i, j], f"{i}-{j}") for i in range(n) for j in range(n)), reverse=True)[:10]
        st.dataframe(pd.DataFrame([(s_, round(p * 100, 2)) for p, s_ in flat], columns=[t("score"), t("pct")]), hide_index=True)
    c = st.columns(5)
    for col, line in zip(c, (0.5, 1.5, 2.5, 3.5, 4.5)): col.markdown(ui.card(f"{t('over')} {line}", f"{over_under(m, line)['over'] * 100:.0f}%"), unsafe_allow_html=True)
    md(ui.card(t("btts"), f"{btts(m) * 100:.0f}%"))

elif page == "odds":
    need_model(); h, a, lh, la = pick_match("o"); st.write(t("enter_odds"))
    c = st.columns(3); oh = c[0].number_input(t("o_home"), 1.01, 100.0, 2.30); od = c[1].number_input(t("o_draw"), 1.01, 100.0, 3.30); oa = c[2].number_input(t("o_away"), 1.01, 100.0, 3.10)
    p = list(result_probs(score_matrix(lh, la, dc.rho)).values()); rows = []
    for r, nm in zip(compare(p, oh, od, oa), (t("o_home"), t("o_draw"), t("o_away"))):
        read = t("model_higher") if r["diff"] > 0.03 else t("model_lower") if r["diff"] < -0.03 else t("model_close")
        rows.append({t("c_outcome"): nm, t("c_model"): round(r["model"] * 100, 1), t("c_market"): round(r["market"] * 100, 1), t("c_diff"): round(r["diff"] * 100, 1),
                     t("c_odds"): r["odds"], t("fair_odds"): round(1 / max(r["model"], 1e-9), 2), t("c_ev"): round(r["ev"], 3), t("c_read"): read})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.write(t("margin", m=overround(oh, od, oa) * 100)); md(ui.warn(t("ev_warn"))); md(ui.disclaimer(t("no_bet")))
    if st.button(t("fetch_odds")):
        try: st.dataframe(OddsApi().h2h_odds())
        except ProviderError as e: st.error(str(e))

elif page == "teams":
    need_model(); tm = st.selectbox(t("team"), dc.teams, format_func=label)
    st.dataframe(stats_table(**{t("col_all"): sv.team_summary(df, tm), t("col_l5"): sv.team_summary(df, tm, 5), t("col_l10"): sv.team_summary(df, tm, 10),
                                t("col_homev"): sv.team_summary(df, tm, venue="home"), t("col_awayv"): sv.team_summary(df, tm, venue="away")}), use_container_width=True)
    i = dc.idx[tm]; st.write(t("strengths", a=dc.att[i], d=dc.dfn[i], h=dc.home))

elif page == "backtest":
    st.write(t("bt_desc"))
    c = st.columns(3); mt = c[0].number_input(t("bt_min"), 100, 5000, 300); stp = c[1].number_input(t("bt_step"), 10, 500, 50); ml_on = c[2].checkbox(t("bt_ml"), True)
    if st.button(t("bt_run")):
        bar = st.progress(0.0); st.session_state.bt = run_backtest(df, int(mt), int(stp), ml_on, bar.progress)
    r = st.session_state.get("bt")
    if r:
        if r.get("n_evaluated"): st.success(t("bt_done"))
        else: st.error(t("bt_few", n=int(mt) + int(stp)))

elif page == "perf":
    r = st.session_state.get("bt")
    if not r or not r.get("n_evaluated"): st.info(t("perf_none"))
    else:
        (st.success if r["validated"] else st.warning)(t("bt_ok") if r["validated"] else t("bt_small", n=r["n_evaluated"]))
        if is_demo: st.error(t("perf_demo"))
        st.subheader(t("perf_h")); st.dataframe(pd.DataFrame(r["1x2"]).T.round(4)); st.caption(t("perf_naive"))
        if "mean_ml_weight" in r: st.caption(t("perf_w", w=r["mean_ml_weight"]))
        c = st.columns(3)
        c[0].write(t("perf_exact")); c[0].json({k: round(v, 4) for k, v in r["exact_score"].items()})
        if "half_time_1x2" in r: c[1].write(t("perf_ht")); c[1].json({k: round(v, 4) for k, v in r["half_time_1x2"].items()})
        if "betman_joint" in r: c[2].write(t("perf_joint")); c[2].json({k: round(v, 4) for k, v in r["betman_joint"].items()})
        fig = px.scatter(r["reliability"], x="predicted", y="observed", size="count", title=t("perf_cal"))
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash")); st.plotly_chart(style(fig), use_container_width=True)

else:
    st.subheader(t("src_keys")); st.code(t("src_steps"), language="text")
    for name in ("API_FOOTBALL_KEY", "ODDS_API_KEY"): st.write(f"**{name}**: {t('src_set') if config.get_key(name) else t('src_unset')}")
    if st.button(t("src_test")):
        try: ApiFootball().status(); st.success(t("src_conn"))
        except ProviderError as e: st.error(str(e))
    st.subheader(t("src_fd"))
    c = st.columns(3); lg = c[0].text_input(t("src_lg"), "E0"); ss = c[1].text_input(t("src_ss"), "2425")
    if c[2].button(t("src_dl")):
        try: n = db.save_matches(fetch_football_data(ss, lg)); st.session_state.ver += 1; get_data.clear(); st.success(t("src_saved", n=n))
        except ProviderError as e: st.error(str(e))
    st.subheader(t("src_up")); up = st.file_uploader(t("src_upl"), type="csv")
    if up and st.button(t("src_save")):
        try: n = db.save_matches(load_csv(up)); st.session_state.ver += 1; get_data.clear(); st.success(t("src_saved", n=n))
        except ProviderError as e: st.error(str(e))
    st.subheader(t("src_api")); c = st.columns(3); lid = c[0].number_input(t("league_id"), 1, 2000, 39); se = c[1].number_input(t("season"), 2015, 2030, 2024)
    if c[2].button(t("src_imp")):
        try: n = db.save_matches(ApiFootball().finished_matches(int(lid), int(se))); get_data.clear(); st.success(t("src_saved2", n=n))
        except ProviderError as e: st.error(str(e))
    st.subheader(t("src_dq")); q = sv.data_quality(df)
    for code, kw in q: md(ui.warn(t(code, **kw)))
    if not q: st.success(t("dq_ok"))
