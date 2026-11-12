"""
Topological Data Analysis of Market Crashes
Replicates Gidea & Katz (2018), "Topological Data Analysis of Financial Time Series:
Landscapes of Crashes" (arxiv.org/abs/1703.04385), then compares the TDA signal
against plain rolling volatility.

Pipeline:
  1. Download daily prices for 4 US indices, compute log returns
  2. Slide a 50-day window -> each window is 50 points in R^4 (a point cloud)
  3. Persistent homology (Vietoris-Rips, H1 = loops) for every window
  4. Turn each persistence diagram into a persistence landscape
  5. Take the L1 and L2 norms of the landscape -> one number per day
  6. Plot the norms around crashes, compare with rolling volatility

Run:
  pip install numpy pandas matplotlib yfinance ripser
  python tda_crashes.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ripser import ripser

trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 2.x vs 1.x

# ---------------- settings ----------------
TICKERS = {"^GSPC": "S&P 500", "^DJI": "Dow Jones", "^IXIC": "Nasdaq", "^RUT": "Russell 2000"}
START = "1987-12-23"        # Russell 2000 data starts around here
END = None                  # None = today
WINDOW = 50                 # days per point cloud (paper uses 50)
N_LAYERS = 5                # how many landscape functions lambda_1..lambda_k to keep
GRID_POINTS = 500           # resolution for integrating the landscape
VOL_WINDOW = 50             # rolling volatility window, same length for a fair comparison
DATA_FILE = "prices.csv"
OUT_DIR = "output"

CRASHES = {
    "Dot-com (2000-03-10)": "2000-03-10",
    "Lehman (2008-09-15)": "2008-09-15",
    "COVID (2020-02-20)": "2020-02-20",
    "2022 bear (2022-01-03)": "2022-01-03",
    "Tariff crash (2025-04-03)": "2025-04-03",
    "Midterm AI selloff (2026-11-05)": "2026-11-05",
}


# ---------------- 1. data ----------------
def load_prices():
    """Download once, then reuse the cached CSV."""
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    import yfinance as yf
    raw = yf.download(list(TICKERS), start=START, end=END, auto_adjust=True, progress=False)
    if raw.empty:
        raise SystemExit("Download failed. Check your internet connection or update yfinance (pip install -U yfinance).")
    prices = raw["Close"][list(TICKERS)].dropna()
    prices.columns = [TICKERS[t] for t in prices.columns]
    prices.to_csv(DATA_FILE)
    return prices


def log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()


# ---------------- 2-4. persistent homology -> landscape ----------------
def h1_diagram(cloud):
    """Vietoris-Rips persistence, return the H1 (loops) birth/death pairs."""
    dgm = ripser(cloud, maxdim=1)["dgms"][1]
    return dgm[np.isfinite(dgm[:, 1])]


def landscape(dgm, grid, k=N_LAYERS):
    """
    Persistence landscape on a grid.
    Each (birth, death) pair makes a tent function: max(0, min(t - b, d - t)).
    lambda_j(t) = j-th largest tent value at t.
    Returns an array of shape (k, len(grid)).
    """
    if len(dgm) == 0:
        return np.zeros((k, len(grid)))
    b, d = dgm[:, 0][:, None], dgm[:, 1][:, None]
    tents = np.maximum(0, np.minimum(grid - b, d - grid))      # (n_pairs, n_grid)
    tents = -np.sort(-tents, axis=0)                            # sort descending per t
    out = np.zeros((k, len(grid)))
    m = min(k, tents.shape[0])
    out[:m] = tents[:m]
    return out


def landscape_norms(lam, grid):
    """L1 and L2 norms, integrated with the trapezoid rule and summed over layers."""
    l1 = trapz(np.abs(lam), grid, axis=1).sum()
    l2 = np.sqrt(trapz(lam ** 2, grid, axis=1).sum())
    return l1, l2


def compute_norms(returns, window=WINDOW):
    X = returns.values
    n = len(X) - window + 1

    # one shared grid so norms are comparable across windows
    # (upper bound = largest pairwise distance in a typical stressed window)
    max_scale = np.percentile(np.abs(X), 99.9) * 2 * np.sqrt(X.shape[1])
    grid = np.linspace(0, max_scale, GRID_POINTS)

    l1s, l2s = np.empty(n), np.empty(n)
    for i in range(n):
        dgm = h1_diagram(X[i:i + window])
        l1s[i], l2s[i] = landscape_norms(landscape(dgm, grid), grid)
        if i % 1000 == 0:
            print(f"  window {i}/{n}")

    idx = returns.index[window - 1:]          # date = last day in the window
    return pd.DataFrame({"L1": l1s, "L2": l2s}, index=idx)


# ---------------- 5-6. analysis + plots ----------------
def plot_crash(norms, vol, name, date, days_before=500, days_after=100):
    d = pd.Timestamp(date)
    lo, hi = d - pd.Timedelta(days=int(days_before * 1.45)), d + pd.Timedelta(days=int(days_after * 1.45))
    sub, v = norms.loc[lo:hi], vol.loc[lo:hi]
    if sub.empty:
        return
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(sub.index, sub["L1"], label="L1 norm", lw=1.2)
    ax1.plot(sub.index, sub["L2"], label="L2 norm", lw=1.2)
    ax1.set_ylabel("landscape norm")
    ax2 = ax1.twinx()
    ax2.plot(v.index, v, color="gray", alpha=0.6, lw=1, label=f"{VOL_WINDOW}d vol (S&P)")
    ax2.set_ylabel("rolling volatility")
    ax1.axvline(d, color="red", ls="--", lw=1)
    ax1.set_title(name)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"crash_{date}.png"), dpi=150)
    plt.close(fig)


def vol_comparison(norms, vol):
    """Is the TDA signal just volatility? Regress L1 on vol and look at what's left."""
    df = pd.concat([norms["L1"], vol.rename("vol")], axis=1, sort=True).dropna()
    corr = df.corr().iloc[0, 1]
    beta, alpha = np.polyfit(df["vol"], df["L1"], 1)
    df["resid"] = df["L1"] - (alpha + beta * df["vol"])
    r2 = 1 - df["resid"].var() / df["L1"].var()
    print(f"\nCorrelation(L1, vol) = {corr:.3f}")
    print(f"R^2 of L1 ~ vol      = {r2:.3f}  (share of the TDA signal explained by volatility)")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df.index, df["resid"], lw=0.8)
    for name, date in CRASHES.items():
        ax.axvline(pd.Timestamp(date), color="red", ls="--", lw=0.8)
    ax.set_title("L1 norm after removing volatility (residual)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "residual_vs_vol.png"), dpi=150)
    plt.close(fig)
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    prices = load_prices()
    rets = log_returns(prices)
    print(f"{len(rets)} trading days, {rets.index[0].date()} to {rets.index[-1].date()}")

    print("Computing persistent homology for every window...")
    norms = compute_norms(rets)
    norms.to_csv(os.path.join(OUT_DIR, "landscape_norms.csv"))

    vol = rets.iloc[:, 0].rolling(VOL_WINDOW).std()

    # full history
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(norms.index, norms["L1"], lw=0.7, label="L1")
    ax.plot(norms.index, norms["L2"], lw=0.7, label="L2")
    for date in CRASHES.values():
        ax.axvline(pd.Timestamp(date), color="red", ls="--", lw=0.8)
    ax.set_title(f"Persistence landscape norms ({WINDOW}-day windows)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "norms_full.png"), dpi=150)
    plt.close(fig)

    for name, date in CRASHES.items():
        plot_crash(norms, vol, name, date)

    vol_comparison(norms, vol)
    print(f"\nDone. Plots and CSV are in ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
