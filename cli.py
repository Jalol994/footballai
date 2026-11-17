"""Command-line runner: python cli.py [--csv file.csv] [--home TEAM --away TEAM]"""
import argparse, sys
from data.demo_data import make_demo
from data.providers import load_csv
from engine.dixon_coles import DixonColes, first_half_share
from engine.betman import betman_joint, marginals, top_combinations, uncertainty, CATS
from engine.backtest import run_backtest

ap = argparse.ArgumentParser(); ap.add_argument("--csv"); ap.add_argument("--home"); ap.add_argument("--away")
ap.add_argument("--backtest", action="store_true"); a = ap.parse_args()
df = load_csv(a.csv) if a.csv else make_demo(seasons=4)
print("DATA:", "DEMO (simulated)" if not a.csv else a.csv, f"| {len(df)} matches")
dc = DixonColes().fit(df); home = a.home or dc.teams[0]; away = a.away or dc.teams[1]
f = first_half_share(df) or 0.45; lh, la = dc.rates(home, away); J = betman_joint(lh, la, f, dc.rho)
print(f"\n{home} vs {away}  xG {lh:.2f}-{la:.2f}  1H share {f:.2f}")
for k, v in marginals(J).items():
    print(f"  {k:8s}", "  ".join(f"{c}:{p*100:4.1f}%" for c, p in zip(CATS, v)), f"| spread {uncertainty(v):.2f}")
print("Top 10 (1H h-a, FT h-a):")
for p, c in top_combinations(J): print(f"  {c}  {p*100:.2f}%")
if a.backtest:
    r = run_backtest(df); print("\n", r["message"])
    for k, v in r.get("1x2", {}).items(): print(f"  {k:40s} acc {v['accuracy']:.3f} brier {v['brier']:.3f} logloss {v['log_loss']:.3f} ece {v['calibration_error']:.3f}")
    for k in ("exact_score", "half_time_1x2", "betman_joint"): print(" ", k, {x: round(y, 4) for x, y in r.get(k, {}).items()})
