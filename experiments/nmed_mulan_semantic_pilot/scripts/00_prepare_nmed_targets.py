from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ExperimentConfig, build_original_group_timeseries, read_nmed_window_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare NMED-E original-condition target time series.")
    parser.add_argument("--window-csv", default="data/external/nmed_window_level_data.csv")
    parser.add_argument("--output-csv", default="outputs/tables/nmed_original_group_timeseries.csv")
    parser.add_argument("--summary-json", default="outputs/tables/nmed_original_group_timeseries_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig()
    window_df = read_nmed_window_table(args.window_csv)
    targets = build_original_group_timeseries(window_df, config=config)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(output_csv, index=False)

    summary = {
        "condition": config.condition,
        "n_windows": int(len(targets)),
        "n_valid_fast60_windows": int(targets["engagement_fast60"].notna().sum()),
        "n_subjects_min": int(targets["n_subjects"].min()),
        "n_subjects_max": int(targets["n_subjects"].max()),
        "window_seconds": config.window_seconds,
        "fast_timescale_seconds": config.fast_timescale_seconds,
    }
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {output_csv}")
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
