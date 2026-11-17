"""Walk-forward backtest: train only on the past, predict the next block, move forward."""
import numpy as np
import pandas as pd
from .score_model import score_matrix, result_probs
from .betman import betman_joint
from .dixon_coles import DixonColes, first_half_share
from .odds import remove_margin
from . import ml

EPS = 1e-12
MIN_FOR_VALIDATION = 1000


def _res(a, b): return 0 if a > b else 1 if a == b else 2


def metrics(P, y):
    """P: (N,3) probabilities, y: (N,) outcome index."""
    P = np.clip(np.asarray(P, float), EPS, 1); P = P / P.sum(1, keepdims=True); y = np.asarray(y)
    onehot = np.eye(3)[y]; conf = P.max(1); correct = (P.argmax(1) == y).astype(float)
    ece = 0.0
    for lo in np.linspace(0, 0.9, 10):          # expected calibration error on top-class confidence
        m = (conf > lo) & (conf <= lo + 0.1)
        if m.any(): ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return dict(n=int(len(y)), accuracy=float(correct.mean()), brier=float(((P - onehot) ** 2).sum(1).mean()),
                log_loss=float(-np.log(P[np.arange(len(y)), y]).mean()), calibration_error=float(ece))


def _best_weight(p_dc, p_ml, y):
    """Blend weight for ML picked from PAST out-of-sample results only (min log loss)."""
    best, bw = 9, 0.0
    for w in np.linspace(0, 1, 11):
        p = np.clip((1 - w) * p_dc + w * p_ml, EPS, 1)
        ll = -np.log(p[np.arange(len(y)), y]).mean()
        if ll < best: best, bw = ll, w
    return float(bw)


def run_backtest(df, min_train=300, step=50, use_ml=True, progress=None):
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) < min_train + step:
        return dict(validated=False, n_evaluated=0, message=f"Need at least {min_train + step} matches; have {len(df)}. Model NOT validated.")
    y_all = ml.outcome(df); X_all = ml.build_features(df) if use_ml else None
    rec = []; past = dict(dc=[], ml=[], y=[]); skipped = 0
    for s in range(min_train, len(df), step):
        train, block = df.iloc[:s], df.iloc[s:s + step]
        dc = DixonColes().fit(train); f = first_half_share(train) or 0.45
        base = np.bincount(y_all[:s], minlength=3) / s
        mdl = ml.fit_calibrated(X_all.iloc[:s], y_all[:s]) if use_ml else None
        w = _best_weight(np.array(past["dc"]), np.array(past["ml"]), np.array(past["y"])) if (mdl is not None and len(past["y"]) >= 100) else 0.0
        for i, r in zip(block.index, block.itertuples()):
            if r.home not in dc.idx or r.away not in dc.idx: skipped += 1; continue
            lh, la = dc.rates(r.home, r.away)
            m = score_matrix(lh, la, dc.rho); p_dc = np.array(list(result_probs(m).values()))
            p_ml = mdl.predict_proba(X_all.iloc[[i]])[0] if mdl is not None else np.full(3, np.nan)
            row = dict(date=r.date, y=y_all[i], p_dc=p_dc, p_ml=p_ml, w=w, p_base=base,
                       p_ens=(1 - w) * p_dc + w * p_ml if mdl is not None else p_dc)
            if mdl is not None: past["dc"].append(p_dc); past["ml"].append(p_ml); past["y"].append(y_all[i])
            if all(pd.notna(getattr(r, c, np.nan)) for c in ("oh", "od", "oa")):
                row["p_book"] = remove_margin(r.oh, r.od, r.oa)
            # exact full-time score
            hg, ag = min(r.hg, 10), min(r.ag, 10)
            row["score_p"] = m[hg, ag]; row["score_top1"] = int(np.unravel_index(m.argmax(), m.shape) == (hg, ag))
            # half-time result
            if pd.notna(getattr(r, "ht_hg", np.nan)) and pd.notna(getattr(r, "ht_ag", np.nan)):
                m1 = score_matrix(lh * f, la * f, dc.rho)
                row["ht_y"] = _res(r.ht_hg, r.ht_ag); row["ht_p"] = np.array(list(result_probs(m1).values()))
                J = betman_joint(lh, la, f, dc.rho)
                key = (min(int(r.ht_hg), 5), min(int(r.ht_ag), 5), min(r.hg, 5), min(r.ag, 5))
                order = np.argsort(J, axis=None)[::-1]
                rank = int(np.where(order == np.ravel_multi_index(key, J.shape))[0][0])
                row.update(joint_p=J[key], joint_top1=int(rank == 0), joint_top10=int(rank < 10))
            rec.append(row)
        if progress: progress(min(1.0, (s + step) / len(df)))
    return summarize(pd.DataFrame(rec), skipped)


def summarize(R, skipped=0):
    n = len(R)
    out = dict(n_evaluated=n, skipped_unseen_teams=skipped, validated=n >= MIN_FOR_VALIDATION, raw=R)
    out["message"] = ("Sample is large enough for a first assessment, but past performance does not guarantee future results."
                      if out["validated"] else f"Only {n} matches evaluated (< {MIN_FOR_VALIDATION}): model NOT validated; treat all numbers as indicative only.")
    if n == 0: return out
    y = R.y.values; stack = lambda c, idx=None: np.vstack(R[c].values if idx is None else R.loc[idx, c].values)
    out["1x2"] = {"Dixon-Coles": metrics(stack("p_dc"), y), "Naive (past frequencies)": metrics(stack("p_base"), y)}
    if R.p_ml.apply(lambda v: not np.isnan(v).any()).any():
        k = R.index[R.p_ml.apply(lambda v: not np.isnan(v).any())]
        out["1x2"]["ML"] = metrics(stack("p_ml", k), y[k]); out["1x2"]["Ensemble"] = metrics(stack("p_ens", k), y[k])
        out["ml_rows"] = int(len(k)); out["mean_ml_weight"] = float(R.loc[k, "w"].mean())
    if "p_book" in R:
        k = R.index[R.p_book.notna()]
        if len(k):
            out["1x2"]["Bookmaker (margin removed)"] = metrics(stack("p_book", k), y[k])
            out["1x2"]["Dixon-Coles (same matches as bookmaker)"] = metrics(stack("p_dc", k), y[k]); out["book_rows"] = int(len(k))
    out["exact_score"] = dict(top1_hit_rate=float(R.score_top1.mean()), mean_log_prob=float(np.log(np.clip(R.score_p, EPS, 1)).mean()))
    if "ht_y" in R and R.ht_y.notna().any():
        k = R.index[R.ht_y.notna()]
        out["half_time_1x2"] = metrics(stack("ht_p", k), R.loc[k, "ht_y"].astype(int).values)
        out["betman_joint"] = dict(n=int(len(k)), top1_hit_rate=float(R.loc[k, "joint_top1"].mean()),
                                   top10_hit_rate=float(R.loc[k, "joint_top10"].mean()),
                                   mean_log_prob=float(np.log(np.clip(R.loc[k, "joint_p"].astype(float), EPS, 1)).mean()))
    P = stack("p_dc"); rel = []                    # reliability table over all classes
    pf, of = P.ravel(), np.eye(3)[y].ravel()
    for lo in np.linspace(0, 0.9, 10):
        m = (pf > lo) & (pf <= lo + 0.1)
        if m.any(): rel.append(dict(bin=f"{lo:.1f}-{lo+.1:.1f}", predicted=float(pf[m].mean()), observed=float(of[m].mean()), count=int(m.sum())))
    out["reliability"] = pd.DataFrame(rel)
    return out
