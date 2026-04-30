"""Recompute hybrid SKU router research artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments_v2.hybrid_router_rules import (  # noqa: E402
    load_router_candidates,
    write_artifacts,
)

DEFAULT_INPUT = (
    ROOT / "reports" / "hybrid_research" / "router_candidates_with_rules.csv"
)
DEFAULT_OUTPUT = ROOT / "reports" / "hybrid_research"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild hybrid router research artifacts"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the benchmark candidate table",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory where the v1 artifacts should be written",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional filename prefix for the generated CSV files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    router_df = load_router_candidates(args.input)
    artifacts = write_artifacts(router_df, args.output_dir, prefix=args.prefix)

    print("Hybrid router artifacts written:")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
