from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export unified metrics CSV to Excel (.xlsx).")
    p.add_argument("--metrics", type=str, required=True, help="Path to metrics.csv")
    p.add_argument("--out", type=str, required=True, help="Output .xlsx path")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    metrics_path = Path(args.metrics)
    out_path = Path(args.out)
    df = pd.read_csv(metrics_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()

