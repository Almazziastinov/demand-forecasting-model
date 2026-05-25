from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "processed" / "kazan_reconstructed_sku_day.csv"
DEFAULT_OUTPUT = ROOT / "reports" / "reconstruction_plots"


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE)
    value = re.sub(r"[\s_-]+", "_", value, flags=re.UNICODE)
    return value.strip("_") or "plot"


def load_reconstruction(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["observed_sales_qty"] = pd.to_numeric(df["observed_sales_qty"], errors="coerce")
    df["reconstructed_sales_qty"] = pd.to_numeric(df["reconstructed_sales_qty"], errors="coerce")
    df["reconstruction_abs_gap"] = pd.to_numeric(df["reconstruction_abs_gap"], errors="coerce")
    return df.dropna(subset=["date"]).copy()


def select_examples(df: pd.DataFrame, examples_per_path: int) -> pd.DataFrame:
    pair_metrics = (
        df.groupby(
            [
                "bakery_id",
                "bakery_name",
                "product_id",
                "product_name",
                "best_decomposition_path",
            ],
            as_index=False,
        )
        .agg(
            mean_abs_gap=("reconstruction_abs_gap", "mean"),
            mean_observed=("observed_sales_qty", "mean"),
            mean_reconstructed=("reconstructed_sales_qty", "mean"),
        )
    )
    pair_metrics["gap_ratio"] = pair_metrics["mean_abs_gap"] / pair_metrics["mean_observed"].replace(0, pd.NA)
    pair_metrics["gap_ratio"] = pd.to_numeric(pair_metrics["gap_ratio"], errors="coerce")

    selected = (
        pair_metrics.sort_values(
            ["best_decomposition_path", "gap_ratio", "mean_observed"],
            ascending=[True, True, False],
        )
        .groupby("best_decomposition_path", as_index=False, group_keys=False)
        .head(examples_per_path)
    )
    return selected


def plot_pair(df: pd.DataFrame, *, bakery_id: str, product_id: str, output_dir: Path) -> Path | None:
    pair = df[
        (df["bakery_id"].astype(str) == str(bakery_id))
        & (df["product_id"].astype(str) == str(product_id))
    ].copy()
    if pair.empty:
        return None

    pair = pair.sort_values("date")
    bakery_name = str(pair["bakery_name"].iloc[0])
    product_name = str(pair["product_name"].iloc[0])
    path_name = str(pair["best_decomposition_path"].iloc[0])

    plt.figure(figsize=(14, 6))
    plt.plot(pair["date"], pair["observed_sales_qty"], label="Observed", linewidth=2.0)
    plt.plot(pair["date"], pair["reconstructed_sales_qty"], label="Reconstructed", linewidth=2.0)
    plt.title(f"{bakery_name} | {product_name}\n{path_name}")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f"{bakery_id}_{product_id}_{_slugify(product_name)}.png"
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def build_plots(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    bakery_id: str | None,
    product_id: str | None,
    examples_per_path: int,
) -> list[Path]:
    df = load_reconstruction(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    if bakery_id is not None and product_id is not None:
        path = plot_pair(df, bakery_id=bakery_id, product_id=product_id, output_dir=out_dir)
        return [path] if path is not None else []

    examples = select_examples(df, examples_per_path=examples_per_path)
    for _, row in examples.iterrows():
        path = plot_pair(
            df,
            bakery_id=str(row["bakery_id"]),
            product_id=str(row["product_id"]),
            output_dir=out_dir,
        )
        if path is not None:
            generated.append(path)
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot observed vs reconstructed sales for Kazan sample")
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--bakery-id")
    parser.add_argument("--product-id")
    parser.add_argument("--examples-per-path", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated = build_plots(
        input_path=args.input_path,
        output_dir=args.output_dir,
        bakery_id=args.bakery_id,
        product_id=args.product_id,
        examples_per_path=args.examples_per_path,
    )
    print("=" * 72)
    print("KAZAN RECONSTRUCTION PLOTS")
    print("=" * 72)
    if not generated:
        print("No plots generated")
        return
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
