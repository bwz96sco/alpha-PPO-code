#!/usr/bin/env python3
"""Summarise legacy Excel logs into a single table.

This script scans a ResultExcel folder, extracts the average value of the "min"
column from the "test" sheet, and writes a `summary_avg_min.xlsx`.

It supports both:
1) "normal" filenames containing `<part>-<mach>-<dist>` (e.g. `GA-50-18-h...`)
2) machine-generalisation filenames like:
   `pure_policy-mach-ex-M18-50-16-h_*.xls`
   where the *test* environment should be interpreted as `50-18-h` (part=50,
   mach=18, dist=h). The `50-16-h` fragment is the *model* environment and is
   ignored for combo grouping.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict
from pathlib import Path

try:
    import pandas as pd
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'pandas'.\n"
        "Tip: run with the project venv: `alpha-schedule-sb3/.venv/bin/python old_ver_code/tmp_scripts/extract_summary_table.py ...`\n"
        "Or install: `pip install pandas openpyxl`."
    ) from e

try:
    import xlrd
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'xlrd' (used for reading .xls files). Install: `pip install xlrd`."
    ) from e


def _extract_min_average_from_xlrd(filepath: str) -> float | None:
    workbook = xlrd.open_workbook(filepath)
    if "test" not in workbook.sheet_names():
        return None
    sheet = workbook.sheet_by_name("test")

    header_row = [str(sheet.cell_value(0, col)).strip().lower() for col in range(sheet.ncols)]
    if "min" in header_row:
        min_col_idx = header_row.index("min")
    elif "finishnum" in header_row:
        min_col_idx = header_row.index("finishnum")
    else:
        return None

    values: list[float] = []
    for row in range(1, sheet.nrows):
        val = sheet.cell_value(row, min_col_idx)
        if isinstance(val, (int, float)):
            values.append(float(val))
    return (sum(values) / len(values)) if values else None


def _extract_min_average_from_pandas(filepath: str) -> float | None:
    try:
        df = pd.read_excel(filepath, sheet_name="test")
    except ValueError:
        return None
    if df.empty:
        return None

    cols = {str(c).strip().lower(): c for c in df.columns}
    if "min" in cols:
        series = df[cols["min"]]
    elif "finishnum" in cols:
        series = df[cols["finishnum"]]
    else:
        return None
    series = pd.to_numeric(series, errors="coerce").dropna()
    return float(series.mean()) if not series.empty else None


def extract_min_average_from_file(filepath: str) -> float | None:
    """Extract average of 'min' column from the 'test' sheet."""
    try:
        return _extract_min_average_from_xlrd(filepath)
    except Exception:
        try:
            return _extract_min_average_from_pandas(filepath)
        except Exception as e:
            print(f"  Error reading {os.path.basename(filepath)}: {e}")
            return None


_RE_MACHINE_GEN = re.compile(
    # e.g. pure_policy-mach-ex-M18-50-16-h_03031715V0.xls  ->  part=50, mach=18, dist=h
    r"(?:^|-)M(?P<test_mach>\d+)-(?P<part>\d+)-(?P<model_mach>\d+)-(?P<dist>[hlm])(?:[_\.-]|$)",
    flags=re.IGNORECASE,
)
_RE_STD = re.compile(r"(?P<part>\d+)-(?P<mach>\d+)-(?P<dist>[hlm])", flags=re.IGNORECASE)


def parse_filename(filename: str) -> tuple[str, str, str] | None:
    """Extract (partNum, machNum, dist) from filename.

    Priority:
    1) machine generalisation pattern: `...-M<mach>-<part>-<model_mach>-<dist>...`
       -> returns (<part>, <mach>, <dist>) using the test mach.
    2) standard pattern: `<part>-<mach>-<dist>`
    """
    m = _RE_MACHINE_GEN.search(filename)
    if m:
        return m.group("part"), m.group("test_mach"), m.group("dist").lower()
    m = _RE_STD.search(filename)
    if m:
        return m.group("part"), m.group("mach"), m.group("dist").lower()
    return None


def infer_algo_name(base_folder: str, filepath: str) -> str:
    rel = os.path.relpath(filepath, base_folder)
    parts = rel.split(os.sep)
    if len(parts) > 1:
        return parts[0]
    filename = parts[0]
    # root files: take the prefix before the first "-"
    return filename.split("-", 1)[0]


def main() -> None:
    p = argparse.ArgumentParser(description="Extract avg(min) from ResultExcel files and build summary tables.")
    p.add_argument(
        "--base-folder",
        type=str,
        default="/Users/zhangbowen/Library/CloudStorage/OneDrive-Personal/Papers/AlphaSchedule/ResultExcel/MachGen",
        help="Folder containing Excel result files (may contain subfolders).",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .xlsx path (default: <base-folder>/summary_avg_min.xlsx).",
    )
    args = p.parse_args()
    base_folder = os.path.expanduser(args.base_folder)

    if not os.path.isdir(base_folder):
        raise FileNotFoundError(base_folder)

    output_path = args.output or os.path.join(base_folder, "summary_avg_min.xlsx")
    output_path = os.path.expanduser(output_path)

    # Collect all data: {dist: {algo: {combo: [avg_values]}}}
    data: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    all_combos: set[tuple[int, int]] = set()

    patterns = [os.path.join(base_folder, "**", "*.xls"), os.path.join(base_folder, "**", "*.xlsx")]
    files: list[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))

    files = sorted({os.path.abspath(f) for f in files})
    files = [f for f in files if os.path.abspath(f) != os.path.abspath(output_path)]

    print(f"Scanning: {base_folder}")
    print(f"Found {len(files)} Excel files.")

    for filepath in files:
        filename = os.path.basename(filepath)
        parsed = parse_filename(filename)
        if parsed is None:
            print(f"  Skipping {filename} (cannot parse partNum-machNum-dist)")
            continue

        part_num, mach_num, dist = parsed
        combo = f"{part_num}-{mach_num}"
        all_combos.add((int(part_num), int(mach_num)))

        algo_name = infer_algo_name(base_folder, filepath)

        avg = extract_min_average_from_file(filepath)
        if avg is not None:
            data[dist][algo_name][combo].append(float(avg))

    sorted_combos = sorted(all_combos)
    combo_labels = [f"{p}-{m}" for p, m in sorted_combos]

    print(f"\nCombinations found: {combo_labels}")
    print(f"Distributions found: {sorted(data.keys())}")

    all_algos = sorted({algo for dist_data in data.values() for algo in dist_data})
    if not all_algos:
        print("No algorithm data extracted; nothing to write.")
        return

    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for dist in ["h", "l", "m"]:
            if dist not in data:
                continue
            dist_data = data[dist]

            rows: list[dict[str, object]] = []
            for algo in all_algos:
                row: dict[str, object] = {"Algorithm": algo}
                for combo_label in combo_labels:
                    values = dist_data.get(algo, {}).get(combo_label, [])
                    row[combo_label] = round(min(values), 2) if values else None
                rows.append(row)

            df = pd.DataFrame(rows)
            df.to_excel(writer, sheet_name=f"dist_{dist}", index=False)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
