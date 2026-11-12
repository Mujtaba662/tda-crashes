"""
Numbers for the write-up. Run after tda_crashes.py.

  python analysis.py

Uses prices.csv and output/landscape_norms.csv. Prints:
  1. How much of the TDA signal volatility explains (R^2)
  2. L1 norm in the 60 days before each crash vs its median over the 2 years before that
  3. Gidea & Katz style trend test: Kendall tau of the rolling variance of L1 over the
     250 days before each crash, ranked against every other date (percentile)
  4. False alarm rate for that test
  5. Lead-lag correlation between L1 and volatility
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from tda_crashes import CRASHES, VOL_WINDOW

PRE_DAYS = 60          # "right before the crash" window
BASE_DAYS = 500        # baseline window before that
TREND_DAYS = 250       # Gidea & Katz use 250 trading days
ALARM_TAU = 0.6

norms = pd.read_csv("output/landscape_norms.csv", index_col=0, parse_dates=True)
prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
L1 = norms["L1"]
vol = np.log(prices.iloc[:, 0]).diff().rolling(VOL_WINDOW).std().reindex(L1.index)

# 1. volatility vs TDA
df = pd.concat([L1, vol.rename("vol")], axis=1, sort=True).dropna()
r = df.corr().iloc[0, 1]
print(f"Correlation(L1, vol) = {r:.3f}   R^2 = {r**2:.3f}")

# 2-4. pre-crash behavior
roll_var = L1.rolling(TREND_DAYS).var()


def trend(d):
    s = roll_var.loc[:d].iloc[-TREND_DAYS:].dropna()
    return kendalltau(np.arange(len(s)), s.values)[0] if len(s) > 50 else np.nan


null = np.array([trend(d) for d in L1.index[2 * TREND_DAYS::20]])
null = null[~np.isnan(null)]

print(f"\n{'Crash':34}{'L1 pre / normal':>16}{'trend tau':>11}{'percentile':>12}")
for name, date in CRASHES.items():
    d = pd.Timestamp(date)
    before = L1.loc[:d]
    ratio = before.iloc[-PRE_DAYS:].mean() / before.iloc[-(PRE_DAYS + BASE_DAYS):-PRE_DAYS].median()
    t = trend(d)
    print(f"{name:34}{ratio:15.1f}x{t:11.2f}{(null < t).mean() * 100:11.0f}%")

print(f"\nFalse alarms: tau > {ALARM_TAU} on {(null > ALARM_TAU).mean() * 100:.0f}% of all dates")

# 5. lead-lag: corr(L1 today, vol k days later)
print("\nLead-lag, corr(L1_t, vol_t+k):")
for k in [-40, -20, 0, 20, 40]:
    print(f"  k = {k:+d}: {df['L1'].corr(df['vol'].shift(-k)):.3f}")
