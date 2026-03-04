from __future__ import annotations

import pandas as pd

def wilcoxon_signed_rank_exact(diffs: list[float]) -> tuple[float, float]:
    diffs = [d for d in diffs if d != 0]
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0

    ranks = pd.Series([abs(d) for d in diffs]).rank(method="average").tolist()
    w_plus = sum(r for d, r in zip(diffs, ranks) if d > 0)
    total = sum(ranks)
    stat = min(w_plus, total - w_plus)

    scaled = [int(round(r * 2)) for r in ranks]  # handle .5 ranks
    total_i = sum(scaled)
    stat_i = int(round(stat * 2))

    count = 0
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += scaled[i]
        if min(s, total_i - s) <= stat_i:
            count += 1
    p = count / (1 << n)  # exact two-sided
    return float(stat), float(p)

xlsx = "/Users/zhangbowen/Library/CloudStorage/OneDrive-Personal/Papers/AlphaSchedule/ResultExcel/summary_avg_min.xlsx"
sheet = "dist_h"  # high-load; change if needed

df = pd.read_excel(xlsx, sheet_name=sheet).set_index("Algorithm")
cols = [c for c in df.columns]
gps_name = "GPSearch"  # change to "GPSearch" if that's the row name in your sheet
gps = df.loc[gps_name, cols].tolist()

for algo in df.index:
    if algo == gps_name:
        continue
    diffs = [a - g for a, g in zip(df.loc[algo, cols].tolist(), gps)]
    stat, p = wilcoxon_signed_rank_exact(diffs)
    print(f"{algo:16s} statistic={stat:>4.1f} p={p:.6f} significant={p < 0.05}")