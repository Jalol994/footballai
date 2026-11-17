"""Poisson + Dixon-Coles score matrix (one period of play)."""
import numpy as np
from scipy.stats import poisson

MAXG = 10  # grid size; probability beyond this is negligible and renormalised


def dc_tau(x, y, lh, la, rho):
    """Dixon-Coles low-score correction factor."""
    if x == 0 and y == 0: return 1 - lh * la * rho
    if x == 0 and y == 1: return 1 + lh * rho
    if x == 1 and y == 0: return 1 + la * rho
    if x == 1 and y == 1: return 1 - rho
    return 1.0


def score_matrix(lh, la, rho=0.0, maxg=MAXG):
    """P[i, j] = P(home scores i, away scores j). rho=0 -> plain Poisson."""
    g = np.arange(maxg + 1)
    m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    for x in (0, 1):
        for y in (0, 1):
            m[x, y] *= dc_tau(x, y, lh, la, rho)
    m = np.clip(m, 0, None)
    return m / m.sum()


def result_probs(m):
    """Home win / draw / away win from a score matrix."""
    return {"home": float(np.tril(m, -1).sum()),
            "draw": float(np.trace(m)),
            "away": float(np.triu(m, 1).sum())}


def over_under(m, line):
    g = np.arange(m.shape[0])
    over = float(m[(g[:, None] + g[None, :]) > line].sum())
    return {"over": over, "under": 1 - over}


def btts(m):
    return float(m[1:, 1:].sum())
