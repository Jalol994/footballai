"""DEMO DATA ONLY: simulated matches for testing. NOT real results."""
import numpy as np, pandas as pd


def make_demo(n_teams=12, seasons=2, seed=7):
    rng = np.random.default_rng(seed)
    teams = [f"Demo FC {chr(65+i)}" for i in range(n_teams)]
    att = rng.normal(0, .25, n_teams); dfn = rng.normal(0, .2, n_teams)
    rows, day = [], pd.Timestamp("2023-08-01")
    for _ in range(seasons * 2):
        for i in range(n_teams):
            for j in range(n_teams):
                if i == j: continue
                lh = np.exp(.25 + .2 + att[i] + dfn[j]); la = np.exp(.2 + att[j] + dfn[i])
                h1, a1 = rng.poisson(lh * .45), rng.poisson(la * .45)
                h2, a2 = rng.poisson(lh * .55), rng.poisson(la * .55)
                rows.append(dict(date=day, home=teams[i], away=teams[j], ht_hg=h1, ht_ag=a1,
                                 hg=h1 + h2, ag=a1 + a2, is_demo=True))
                day += pd.Timedelta(hours=14)
    return pd.DataFrame(rows)
