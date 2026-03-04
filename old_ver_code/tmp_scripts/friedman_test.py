from __future__ import annotations

import pandas as pd
import mpmath as mp

def chi2_sf(chi2: float, df: int) -> float:
    s = mp.mpf(df) / 2
    x = mp.mpf(chi2) / 2
    return float(mp.gammainc(s, x, mp.inf) / mp.gamma(s))

xlsx = "/Users/zhangbowen/Library/CloudStorage/OneDrive-Personal/Papers/AlphaSchedule/ResultExcel/summary_avg_min.xlsx"
sheet = "dist_h"  # high-load (change to dist_m / dist_l if needed)

df = pd.read_excel(xlsx, sheet_name=sheet)
value_cols = [c for c in df.columns if c != "Algorithm"]

# blocks x algorithms (each block is one (N,M) column)
values = df.set_index("Algorithm")[value_cols].T.dropna(axis=0, how="any")

ranks = values.rank(axis=1, method="average", ascending=True)  # lower TWT = better rank
N, k = ranks.shape
R = ranks.sum(axis=0)

chi2 = (12.0 / (N * k * (k + 1.0))) * float((R**2).sum()) - 3.0 * N * (k + 1.0)
p = chi2_sf(chi2, df=k - 1)

mean_ranks = ranks.mean(axis=0).sort_values()

print(f"N={N}, k={k}")
print(f"Friedman chi2={chi2:.3f}, df={k-1}, p={p:.3e}")
print(mean_ranks.to_string())