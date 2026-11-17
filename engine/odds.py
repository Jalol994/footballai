"""Odds maths. Informational only: positive EV never guarantees profit."""
import numpy as np


def implied(oh, od, oa):
    """Raw implied probabilities (sum > 1 because of bookmaker margin)."""
    return np.array([1 / oh, 1 / od, 1 / oa])


def overround(oh, od, oa):
    return float(implied(oh, od, oa).sum() - 1)


def remove_margin(oh, od, oa):
    """Proportional margin removal -> probabilities that sum to 1."""
    p = implied(oh, od, oa)
    return p / p.sum()


def expected_value(prob, odds):
    """EV per 1 unit staked, assuming the model probability were exactly right (it is only an estimate)."""
    return prob * odds - 1


def compare(model_p, oh, od, oa):
    mk = remove_margin(oh, od, oa); odds = [oh, od, oa]
    return [dict(outcome=n, model=float(model_p[i]), market=float(mk[i]), diff=float(model_p[i] - mk[i]),
                 odds=odds[i], ev=float(expected_value(model_p[i], odds[i])))
            for i, n in enumerate(["Home", "Draw", "Away"])]
