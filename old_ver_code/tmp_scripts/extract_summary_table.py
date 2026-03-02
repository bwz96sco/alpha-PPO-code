#!/usr/bin/env python3
"""Extract average of 'min' column from Excel files and produce summary tables."""
from __future__ import annotations

import os
import re
import glob
from collections import defaultdict

import xlrd
import pandas as pd


def extract_min_average_from_file(filepath: str) -> float | None:
    """Extract average of 'min' column from the 'test' sheet."""
    try:
        workbook = xlrd.open_workbook(filepath)
        if "test" not in workbook.sheet_names():
            return None
        sheet = workbook.sheet_by_name("test")

        header_row = [
            str(sheet.cell_value(0, col)).lower() for col in range(sheet.ncols)
        ]
        if "min" in header_row:
            min_col_idx = header_row.index("min")
        elif "finishnum" in header_row:
            min_col_idx = header_row.index("finishnum")
        else:
            return None

        min_values = []
        for row in range(1, sheet.nrows):
            val = sheet.cell_value(row, min_col_idx)
            if isinstance(val, (int, float)) and val != "":
                min_values.append(val)

        if min_values:
            return sum(min_values) / len(min_values)
        return None
    except Exception as e:
        print(f"  Error reading {os.path.basename(filepath)}: {e}")
        return None


def parse_filename(filename: str) -> tuple[str, str, str] | None:
    """Extract (partNum, machNum, dist) from filename.

    Looks for pattern like 100-16-h or 50-8-l in the filename.
    """
    match = re.search(r"(\d+)-(\d+)-([hlm])", filename)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None


def main() -> None:
    base_folder = "/Users/zhangbowen/Library/CloudStorage/OneDrive-Personal/Papers/AlphaSchedule/ResultExcel"

    # Collect all data: {dist: {algo: {combo: [avg_values]}}}
    data: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    all_combos: set[tuple[int, int]] = set()

    # Walk each algo subfolder
    for algo_dir in sorted(os.listdir(base_folder)):
        algo_path = os.path.join(base_folder, algo_dir)
        if not os.path.isdir(algo_path):
            continue

        algo_name = algo_dir  # Use folder name as display name
        files = glob.glob(os.path.join(algo_path, "*.xls")) + glob.glob(
            os.path.join(algo_path, "*.xlsx")
        )

        print(f"\nProcessing {algo_name} ({len(files)} files)...")

        for filepath in sorted(files):
            filename = os.path.basename(filepath)
            parsed = parse_filename(filename)
            if parsed is None:
                print(f"  Skipping {filename} (cannot parse partNum-machNum-dist)")
                continue

            part_num, mach_num, dist = parsed
            combo = f"{part_num}-{mach_num}"
            all_combos.add((int(part_num), int(mach_num)))

            avg = extract_min_average_from_file(filepath)
            if avg is not None:
                data[dist][algo_name][combo].append(avg)

    # Sort combos: first by partNum, then by machNum
    sorted_combos = sorted(all_combos)
    combo_labels = [f"{p}-{m}" for p, m in sorted_combos]

    print(f"\nCombinations found: {combo_labels}")
    print(f"Distributions found: {sorted(data.keys())}")

    # Collect all algo names across all dists (for consistent ordering)
    all_algos = sorted({algo for dist_data in data.values() for algo in dist_data})

    # Build output Excel with 3 sheets
    output_path = os.path.join(base_folder, "summary_avg_min.xlsx")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for dist in ["h", "l", "m"]:
            if dist not in data:
                print(f"\nNo data for dist={dist}, skipping.")
                continue

            dist_data = data[dist]
            rows = []
            for algo in all_algos:
                row = {"Algorithm": algo}
                for combo_label in combo_labels:
                    values = dist_data.get(algo, {}).get(combo_label, [])
                    if values:
                        # For GA (or any algo with multiple configs), pick minimum average
                        best = min(values)
                        row[combo_label] = round(best, 2)
                    else:
                        row[combo_label] = None
                rows.append(row)

            df = pd.DataFrame(rows)
            df.to_excel(writer, sheet_name=f"dist_{dist}", index=False)

            print(f"\n=== Distribution: {dist} ===")
            print(df.to_string(index=False))

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
