"""ML 1X2 model (XGBoost if installed, otherwise scikit-learn). Features use ONLY earlier matches."""
import numpy as np
import pandas as pd
from collections import defaultdict

FEATURES = ["h_gf", "h_ga", "h_pts", "a_gf", "a_ga", "a_pts", "h_home_gf", "a_away_gf", "h_rest", "a_rest", "h_n", "a_n"]


def build_features(df, form=5):
    """One row per match (same order as df, which must be date-sorted). No look-ahead:
    features are computed from history BEFORE the match, then the match is added."""
    hist = defaultdict(list)   # team -> list of (date, gf, ga, pts, was_home)
    rows = []
    for r in df.itertuples():
        f = {}
        for side, team in (("h", r.home), ("a", r.away)):
            H = hist[team][-form:]
            f[f"{side}_n"] = len(hist[team])
            f[f"{side}_gf"] = np.mean([x[1] for x in H]) if H else np.nan
            f[f"{side}_ga"] = np.mean([x[2] for x in H]) if H else np.nan
            f[f"{side}_pts"] = np.mean([x[3] for x in H]) if H else np.nan
            f[f"{side}_rest"] = min((r.date - hist[team][-1][0]).days, 30) if hist[team] else np.nan
        hh = [x for x in hist[r.home] if x[4]][-form:]; aa = [x for x in hist[r.away] if not x[4]][-form:]
        f["h_home_gf"] = np.mean([x[1] for x in hh]) if hh else np.nan
        f["a_away_gf"] = np.mean([x[1] for x in aa]) if aa else np.nan
        rows.append(f)
        pts_h = 3 if r.hg > r.ag else 1 if r.hg == r.ag else 0
        hist[r.home].append((r.date, r.hg, r.ag, pts_h, True))
        hist[r.away].append((r.date, r.ag, r.hg, 3 if pts_h == 0 else 1 if pts_h == 1 else 0, False))
    return pd.DataFrame(rows)[FEATURES]


def outcome(df):
    """0 = home win, 1 = draw, 2 = away win."""
    return np.where(df.hg > df.ag, 0, np.where(df.hg == df.ag, 1, 2))


def make_model():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8,
                             eval_metric="mlogloss", verbosity=0), "xgboost"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, random_state=0), "sklearn-hgb"


def fit_calibrated(X, y):
    """Fit + probability calibration (sigmoid). Returns model or None if data is unsuitable."""
    if len(y) < 200 or len(set(y)) < 3:
        return None
    from sklearn.calibration import CalibratedClassifierCV
    model, _ = make_model()
    try:
        cal = CalibratedClassifierCV(model, method="sigmoid", cv=3)
        return cal.fit(X, y)
    except Exception:
        return model.fit(X, y)
