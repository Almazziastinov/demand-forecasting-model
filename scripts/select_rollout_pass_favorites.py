from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = (
    REPO_ROOT
    / "reports"
    / "rollout_sku_risk_audit_runner_city_prior_soft_weekpart"
    / "bakery_sku_risk_summary.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "rollout_pass_favorites"
DEFAULT_EXCLUDED_BAKERY_IDS = "30,105,60"


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def add_favorite_score(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["favorite_score"] = (
        _num(work["sku_wmape_scaled_pct"])
        + 0.70 * _num(work["runner_wmape_scaled_pct"])
        + 0.80 * _num(work["wmape"])
        + 0.50 * _num(work["bias_pct_of_actual_mean"]).abs()
        + 100.0 * _num(work["sku_share_distance"])
        + 80.0 * _num(work["category_share_distance"])
        + 20.0 * _num(work["forecast_only_forecast_share_pct"])
        + 20.0 * _num(work["fact_only_fact_share_pct"])
        + 5.0 * _num(work["eclair_forecast_share_pct"])
        + 3.0 * _num(work["service_forecast_share_pct"])
    )
    work["favorite_score"] = work["favorite_score"].round(3)
    return work


def select_favorites(
    summary: pd.DataFrame,
    *,
    city: str | None,
    min_actual_mean: float,
    min_runner_sku_count: int,
    excluded_bakery_ids: set[int],
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    passed = add_favorite_score(summary[summary["risk_level"].eq("pass")].copy())
    passed["manual_excluded"] = (
        pd.to_numeric(passed["bakery_id"], errors="coerce")
        .fillna(-1)
        .astype("int64")
        .isin(excluded_bakery_ids)
    )
    eligible = passed[
        (_num(passed["actual_mean"]) >= min_actual_mean)
        & (_num(passed["runner_sku_count"]) >= min_runner_sku_count)
        & ~passed["manual_excluded"]
    ].copy()
    if city:
        eligible = eligible[eligible["city"].fillna("").astype(str).eq(city)].copy()

    favorites = eligible.sort_values(
        [
            "favorite_score",
            "sku_wmape_scaled_pct",
            "runner_wmape_scaled_pct",
            "wmape",
            "actual_mean",
        ],
        ascending=[True, True, True, True, False],
    ).head(top_n)
    return passed, favorites


def build_markdown(
    *,
    summary_counts: dict,
    city_counts: pd.Series,
    favorites: pd.DataFrame,
    city: str | None,
    min_actual_mean: float,
    min_runner_sku_count: int,
    excluded_bakery_ids: set[int],
) -> str:
    title_city = city or "all cities"
    lines = [
        "# Rollout Pass Favorites",
        "",
        "Input pass filter: `risk_level == pass`; "
        f"favorite city scope: `{title_city}`.",
        f"Eligibility: `actual_mean >= {min_actual_mean:g}`, "
        f"`runner_sku_count >= {min_runner_sku_count}`.",
        "Manual exclusions: "
        + (", ".join(str(item) for item in sorted(excluded_bakery_ids)) or "none")
        + ".",
        "",
        "## Pass Counts",
        "",
        pd.DataFrame([summary_counts]).to_markdown(index=False),
        "",
        "## Pass By City",
        "",
        city_counts.rename_axis("city").reset_index(name="pass_bakeries").to_markdown(
            index=False,
        ),
        "",
        "## Favorites",
        "",
    ]
    cols = [
        "bakery_id",
        "bakery_name",
        "city",
        "favorite_score",
        "actual_mean",
        "wmape",
        "bias_pct_of_actual_mean",
        "sku_wmape_scaled_pct",
        "runner_wmape_scaled_pct",
        "sku_share_distance",
        "runner_sku_count",
        "eclair_forecast_share_pct",
    ]
    lines.append(favorites[cols].to_markdown(index=False, floatfmt=".2f"))
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict:
    summary = pd.read_csv(args.input)
    excluded_bakery_ids = {
        int(item.strip())
        for item in str(args.exclude_bakery_ids).split(",")
        if item.strip()
    }
    passed, favorites = select_favorites(
        summary,
        city=args.city,
        min_actual_mean=args.min_actual_mean,
        min_runner_sku_count=args.min_runner_sku_count,
        excluded_bakery_ids=excluded_bakery_ids,
        top_n=args.top_n,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pass_path = output_dir / "pass_bakeries.csv"
    favorites_path = output_dir / "pass_favorites.csv"
    markdown_path = output_dir / "pass_favorites.md"
    json_path = output_dir / "pass_favorites_summary.json"

    passed.to_csv(pass_path, index=False, encoding="utf-8-sig")
    favorites.to_csv(favorites_path, index=False, encoding="utf-8-sig")

    city_counts = passed["city"].value_counts()
    summary_counts = {
        "total_bakeries": int(len(summary)),
        "pass_bakeries": int(len(passed)),
        "eligible_favorites": int(
            len(
                passed[
                    (_num(passed["actual_mean"]) >= args.min_actual_mean)
                    & (_num(passed["runner_sku_count"]) >= args.min_runner_sku_count)
                    & ~passed["manual_excluded"]
                    & (
                        True
                        if not args.city
                        else passed["city"].fillna("").astype(str).eq(args.city)
                    )
                ]
            )
        ),
        "selected_favorites": int(len(favorites)),
        "city_scope": args.city or "all",
    }
    markdown_path.write_text(
        build_markdown(
            summary_counts=summary_counts,
            city_counts=city_counts,
            favorites=favorites,
            city=args.city,
            min_actual_mean=args.min_actual_mean,
            min_runner_sku_count=args.min_runner_sku_count,
            excluded_bakery_ids=excluded_bakery_ids,
        ),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(summary_counts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        **summary_counts,
        "pass_path": str(pass_path),
        "favorites_path": str(favorites_path),
        "markdown_path": str(markdown_path),
        "summary_path": str(json_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select rollout pass favorites")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--city", default="Казань")
    parser.add_argument("--min-actual-mean", type=float, default=1000.0)
    parser.add_argument("--min-runner-sku-count", type=int, default=3)
    parser.add_argument("--exclude-bakery-ids", default=DEFAULT_EXCLUDED_BAKERY_IDS)
    parser.add_argument("--top-n", type=int, default=15)
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
