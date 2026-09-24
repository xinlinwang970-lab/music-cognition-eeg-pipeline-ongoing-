from __future__ import annotations

import argparse
from pathlib import Path

from common import ExperimentConfig, build_time_feature_table, validate_feature_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create time-only baseline features for NMED-E original windows.")
    parser.add_argument("--output-csv", default="outputs/features/original_time_features.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig()
    features = build_time_feature_table(config)
    validate_feature_table(features, config)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
