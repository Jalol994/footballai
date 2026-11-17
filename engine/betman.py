"""Joint BETMAN four-category model.

Each half is its own Poisson/Dixon-Coles score matrix:
  1H goals: (H1, A1) ~ matrix(lh*f,     la*f)
  2H goals: (H2, A2) ~ matrix(lh*(1-f), la*(1-f))
  Full time: HT = H1 + H2, AT = A1 + A2
Joint P(h1,a1,h2,a2) = P1(h1,a1) * P2(h2,a2). Full time is a SUM, so it can
never be below half time. Goals are then binned into 0,1,2,3,4,5+.
Approximation: the two halves are treated as independent given team rates.
"""
import numpy as np
from .score_model import score_matrix, MAXG

CATS = ["0", "1", "2", "3", "4", "5+"]


def _bin(n):
    return min(int(n), 5)


def betman_joint(lh, la, first_half_share=0.45, rho=-0.05, maxg=MAXG):
    """Returns J[ht_home, ht_away, ft_home, ft_away], shape (6,6,6,6)."""
    f = first_half_share
    m1 = score_matrix(lh * f, la * f, rho, maxg)
    m2 = score_matrix(lh * (1 - f), la * (1 - f), rho, maxg)
    P = m1[:, :, None, None] * m2[None, None, :, :]          # [h1, a1, h2, a2]
    g = np.arange(maxg + 1)
    b = lambda x: np.minimum(x, 5)
    I0 = b(g)[:, None, None, None] * np.ones_like(P, dtype=int)   # HT home bin
    I1 = b(g)[None, :, None, None] * np.ones_like(P, dtype=int)   # HT away bin
    I2 = b(g[:, None, None, None] + g[None, None, :, None]) * np.ones_like(P, dtype=int)
    I3 = b(g[None, :, None, None] + g[None, None, None, :]) * np.ones_like(P, dtype=int)
    J = np.zeros((6, 6, 6, 6))
    np.add.at(J, (I0, I1, I2, I3), P)
    return J / J.sum()


def marginals(J):
    names = ["ht_home", "ht_away", "ft_home", "ft_away"]
    return {n: J.sum(axis=tuple(k for k in range(4) if k != i)) for i, n in enumerate(names)}


def top_combinations(J, n=10):
    """List of (probability, 'ht_h-ht_a-ft_h-ft_a')."""
    flat = [(float(J[i]), "-".join(CATS[k] for k in i)) for i in np.ndindex(J.shape)]
    flat.sort(reverse=True)
    return flat[:n]


def uncertainty(p):
    """Normalised Shannon entropy of a distribution: 0 = concentrated, 1 = spread out.
    Measures spread of the model's output, NOT validated accuracy."""
    p = np.asarray(p).ravel(); p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(6))


def expected_score(J):
    """Mean goals per variable (5+ counted as 5, so a slight underestimate)."""
    return {k: float((v * np.arange(6)).sum()) for k, v in marginals(J).items()}
