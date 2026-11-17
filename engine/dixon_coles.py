"""Fit Dixon-Coles team strengths with time decay (max likelihood)."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .score_model import dc_tau


class DixonColes:
    def __init__(self, xi=0.0018):  # decay per day (half-life ~ 385 days)
        self.xi = xi

    def fit(self, df, as_of=None):
        """df columns: date, home, away, hg, ag (full-time goals).
        Only matches on/before as_of are used (no look-ahead)."""
        df = df.copy(); df["date"] = pd.to_datetime(df["date"])
        as_of = pd.to_datetime(as_of) if as_of is not None else df["date"].max()
        df = df[df["date"] <= as_of]
        if len(df) < 50:
            raise ValueError(f"Only {len(df)} matches: too few to fit reliably (need >= 50).")
        self.teams = sorted(set(df.home) | set(df.away)); T = len(self.teams)
        self.idx = {t: i for i, t in enumerate(self.teams)}
        h = df.home.map(self.idx).values; a = df.away.map(self.idx).values
        hg = df.hg.values; ag = df.ag.values
        w = np.exp(-self.xi * (as_of - df.date).dt.days.values)

        def nll(p):
            att, dfn, home, rho = p[:T] - p[:T].mean(), p[T:2*T], p[2*T], p[2*T+1]
            lh = np.exp(att[h] + dfn[a] + home); la = np.exp(att[a] + dfn[h])
            tau = np.ones(len(hg))
            tau[(hg == 0) & (ag == 0)] = 1 - lh[(hg == 0) & (ag == 0)] * la[(hg == 0) & (ag == 0)] * rho
            m = (hg == 0) & (ag == 1); tau[m] = 1 + lh[m] * rho
            m = (hg == 1) & (ag == 0); tau[m] = 1 + la[m] * rho
            m = (hg == 1) & (ag == 1); tau[m] = 1 - rho
            if np.any(tau <= 0): return 1e10
            ll = np.log(tau) - lh + hg * np.log(lh) - la + ag * np.log(la)
            return -(w * ll).sum()

        p0 = np.concatenate([np.zeros(2*T), [0.25, -0.05]])
        r = minimize(nll, p0, method="L-BFGS-B",
                     bounds=[(-3, 3)] * (2*T) + [(-1, 1), (-0.3, 0.3)])
        self.att = r.x[:T] - r.x[:T].mean(); self.dfn = r.x[T:2*T]
        self.home, self.rho = float(r.x[2*T]), float(r.x[2*T+1])
        self.n_matches = len(df); self.converged = bool(r.success)
        return self

    def rates(self, home, away):
        """Expected goals (home, away) for a fixture."""
        i, j = self.idx[home], self.idx[away]
        return (float(np.exp(self.att[i] + self.dfn[j] + self.home)),
                float(np.exp(self.att[j] + self.dfn[i])))


def first_half_share(df):
    """Observed share of goals scored in 1H; None if data is insufficient."""
    d = df.dropna(subset=["ht_hg", "ht_ag"])
    tot = (d.hg + d.ag).sum()
    if len(d) < 100 or tot == 0:
        return None
    return float((d.ht_hg + d.ht_ag).sum() / tot)
