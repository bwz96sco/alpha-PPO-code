#!/usr/bin/env python3
"""Extract average of 'min' column from 'test' sheet in Excel files."""

import os
import glob
import xlrd
import pandas as pd

def extract_min_average(folder_path):
    """Extract average of 'min' column from all Excel files in folder."""
    # Find all .xls files in the folder
    xls_pattern = os.path.join(folder_path, "*.xls")
    xlsx_pattern = os.path.join(folder_path, "*.xlsx")

    files = glob.glob(xls_pattern) + glob.glob(xlsx_pattern)

    if not files:
        print(f"No Excel files found in {folder_path}")
        return

    print(f"Found {len(files)} Excel file(s)\n")
    print(f"{'File':<60} {'Avg Min':>12}")
    print("-" * 74)

    all_averages = []

    for filepath in sorted(files):
        filename = os.path.basename(filepath)
        try:
            if filepath.endswith('.xls'):
                # Use xlrd for .xls files
                workbook = xlrd.open_workbook(filepath)
                if 'test' not in workbook.sheet_names():
                    print(f"{filename:<60} {'No test sheet':>12}")
                    continue
                sheet = workbook.sheet_by_name('test')

                # Find 'min' or 'finishnum' column index
                header_row = [str(sheet.cell_value(0, col)).lower() for col in range(sheet.ncols)]
                if 'min' in header_row:
                    min_col_idx = header_row.index('min')
                elif 'finishnum' in header_row:
                    min_col_idx = header_row.index('finishnum')
                else:
                    print(f"{filename:<60} {'No min/finishNum':>12}")
                    continue

                # Extract values (skip header)
                min_values = []
                for row in range(1, sheet.nrows):
                    val = sheet.cell_value(row, min_col_idx)
                    if isinstance(val, (int, float)):
                        min_values.append(val)
            else:
                # Use pandas for .xlsx files
                df = pd.read_excel(filepath, sheet_name='test')
                df.columns = df.columns.str.lower()
                if 'min' in df.columns:
                    min_values = df['min'].dropna().tolist()
                elif 'finishnum' in df.columns:
                    min_values = df['finishnum'].dropna().tolist()
                else:
                    print(f"{filename:<60} {'No min/finishNum':>12}")
                    continue

            if min_values:
                avg = sum(min_values) / len(min_values)
                all_averages.append(avg)
                print(f"{filename:<60} {avg:>12.2f}")
            else:
                print(f"{filename:<60} {'No values':>12}")

        except Exception as e:
            print(f"{filename:<60} Error: {e}")

    if all_averages:
        print("-" * 74)
        print(f"{'Overall Average':<60} {sum(all_averages)/len(all_averages):>12.2f}")
        print(f"{'Total files processed':<60} {len(all_averages):>12}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = "/Users/zhangbowen/Library/CloudStorage/OneDrive-Personal/Papers/AlphaSchedule/LogsNew/Rule"

    extract_min_average(folder)
