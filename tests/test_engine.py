import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from engine.betman import betman_joint, marginals, top_combinations
from engine.score_model import score_matrix, result_probs
from engine.dixon_coles import DixonColes, first_half_share
from data.demo_data import make_demo


def test_matrix_sums_to_one():
    assert abs(score_matrix(1.5, 1.1, -.05).sum() - 1) < 1e-9
    assert abs(sum(result_probs(score_matrix(1.5, 1.1)).values()) - 1) < 1e-9

def test_joint_valid():
    J = betman_joint(1.6, 1.1)
    assert abs(J.sum() - 1) < 1e-9
    for v in marginals(J).values(): assert abs(v.sum() - 1) < 1e-9

def test_ft_never_below_ht():
    J = betman_joint(1.6, 1.1)
    for h, a, H, A in np.ndindex(J.shape):
        if H < h or A < a: assert J[h, a, H, A] == 0

def test_fit_and_top10():
    df = make_demo(); dc = DixonColes().fit(df)
    lh, la = dc.rates(dc.teams[0], dc.teams[1])
    assert dc.converged and 0.3 < lh < 4 and 0.3 < la < 4
    assert 0.35 < first_half_share(df) < 0.55
    assert len(top_combinations(betman_joint(lh, la, first_half_share(df), dc.rho))) == 10

def test_joint_matches_monte_carlo():
    """Independent check: simulate halves directly and compare combo frequencies (rho=0)."""
    rng = np.random.default_rng(1); N = 400000; lh, la, f = 1.6, 1.1, 0.45
    h1, a1 = rng.poisson(lh*f, N), rng.poisson(la*f, N); h2, a2 = rng.poisson(lh*(1-f), N), rng.poisson(la*(1-f), N)
    b = lambda x: np.minimum(x, 5); J = betman_joint(lh, la, f, rho=0.0)
    emp = np.zeros((6, 6, 6, 6)); np.add.at(emp, (b(h1), b(a1), b(h1+h2), b(a1+a2)), 1); emp /= N
    assert np.abs(emp - J).max() < 0.003

def test_odds():
    from engine.odds import remove_margin, overround
    assert abs(remove_margin(2.0, 3.5, 4.0).sum() - 1) < 1e-12 and overround(2.0, 3.5, 4.0) > 0

def test_db_and_csv(tmp_path="/tmp/t.db"):
    import os, pandas as pd
    from data import db
    from data.providers import normalize_football_data
    if os.path.exists(tmp_path): os.remove(tmp_path)
    raw = pd.DataFrame(dict(Date=["12/08/2023"], HomeTeam=["A"], AwayTeam=["B"], FTHG=[2], FTAG=[1], HTHG=[1], HTAG=[0], B365H=[2.0], B365D=[3.4], B365A=[3.9]))
    assert db.save_matches(normalize_football_data(raw), tmp_path) == 1
    assert db.load_matches(tmp_path).iloc[0].ht_hg == 1

def test_missing_key_is_clear_error():
    from data.providers import ApiFootball, ProviderError
    os.environ.pop("API_FOOTBALL_KEY", None)
    try: ApiFootball().status(); assert False
    except ProviderError as e: assert "not set" in str(e)

def test_backtest_runs_and_flags_unvalidated():
    from engine.backtest import run_backtest
    r = run_backtest(make_demo(seasons=2), min_train=200, step=100, use_ml=False)
    assert r["n_evaluated"] > 0 and not r["validated"] and "NOT validated" in r["message"]

def test_services():
    import services as sv
    df = make_demo(); t = df.home.iloc[0]
    assert sv.team_summary(df, t, 5)["matches"] == 5 and len(sv.league_table(df)) == 12 and sv.data_quality(df)

if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
