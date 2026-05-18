from __future__ import annotations

import argparse
import json
from pathlib import Path

import clickhouse_connect
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = ROOT / ".env"
RUNS_TABLE = "forecast_runs_embedded"
BAKERY_DAY_TABLE = "bakery_forecast_day_embedded"
SKU_DAY_TABLE = "sku_forecast_day_embedded"


def load_env_file(path: str | Path = DEFAULT_ENV_PATH) -> dict[str, str]:
    env: dict[str, str] = {}
    file_path = Path(path)
    if not file_path.exists():
        return env

    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def create_client(env_path: str | Path):
    env = load_env_file(env_path)
    return clickhouse_connect.get_client(
        host=env["HOST"],
        port=int(env["PORT"]),
        username=env["USER"],
        password=env["PASSWORD"],
        database=env["DATABASE"],
    )


def get_latest_run_row(client, run_id: str) -> dict | None:
    df = client.query_df(
        f"""
        select run_id, model_version, profile_version, source_kind,
               horizon_start, horizon_end, generated_at, status, notes, is_bias_adjusted
        from {RUNS_TABLE}
        where run_id = %(run_id)s
        order by generated_at desc
        limit 1
        """,
        parameters={"run_id": run_id},
    )
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def get_active_run_id(client) -> str | None:
    df = client.query_df(
        f"""
        select run_id
        from {RUNS_TABLE}
        where status = 'active'
        order by generated_at desc
        limit 1
        """
    )
    if df.empty:
        return None
    return str(df.iloc[0]["run_id"])


def compare_bakery_totals(client, base_run_id: str, candidate_run_id: str) -> pd.DataFrame:
    query = f"""
    with
    base as (
        select forecast_date, bakery_id, bakery_name, city, forecast_final
        from {BAKERY_DAY_TABLE}
        where run_id = %(base_run_id)s
    ),
    candidate as (
        select forecast_date, bakery_id, bakery_name, city, forecast_final
        from {BAKERY_DAY_TABLE}
        where run_id = %(candidate_run_id)s
    )
    select
        coalesce(candidate.forecast_date, base.forecast_date) as forecast_date,
        coalesce(candidate.bakery_id, base.bakery_id) as bakery_id,
        coalesce(candidate.bakery_name, base.bakery_name) as bakery_name,
        coalesce(candidate.city, base.city) as city,
        base.forecast_final as base_forecast,
        candidate.forecast_final as candidate_forecast,
        coalesce(candidate.forecast_final, 0.0) - coalesce(base.forecast_final, 0.0) as delta
    from base
    full outer join candidate
      on candidate.forecast_date = base.forecast_date
     and candidate.bakery_id = base.bakery_id
    """
    return client.query_df(
        query,
        parameters={
            "base_run_id": base_run_id,
            "candidate_run_id": candidate_run_id,
        },
    )


def compare_sku_totals(client, base_run_id: str, candidate_run_id: str) -> pd.DataFrame:
    query = f"""
    with
    base as (
        select forecast_date, bakery_id, product_id, product_name, category_name, forecast_qty
        from {SKU_DAY_TABLE}
        where run_id = %(base_run_id)s
    ),
    candidate as (
        select forecast_date, bakery_id, product_id, product_name, category_name, forecast_qty
        from {SKU_DAY_TABLE}
        where run_id = %(candidate_run_id)s
    )
    select
        coalesce(candidate.forecast_date, base.forecast_date) as forecast_date,
        coalesce(candidate.bakery_id, base.bakery_id) as bakery_id,
        coalesce(candidate.product_id, base.product_id) as product_id,
        coalesce(candidate.product_name, base.product_name) as product_name,
        coalesce(candidate.category_name, base.category_name) as category_name,
        base.forecast_qty as base_forecast,
        candidate.forecast_qty as candidate_forecast,
        coalesce(candidate.forecast_qty, 0.0) - coalesce(base.forecast_qty, 0.0) as delta
    from base
    full outer join candidate
      on candidate.forecast_date = base.forecast_date
     and candidate.bakery_id = base.bakery_id
     and candidate.product_id = base.product_id
    """
    return client.query_df(
        query,
        parameters={
            "base_run_id": base_run_id,
            "candidate_run_id": candidate_run_id,
        },
    )


def build_summary(
    base_run: dict,
    candidate_run: dict,
    bakery_diff: pd.DataFrame,
    sku_diff: pd.DataFrame,
) -> dict:
    bakery_abs = bakery_diff["delta"].abs() if not bakery_diff.empty else pd.Series(dtype=float)
    sku_abs = sku_diff["delta"].abs() if not sku_diff.empty else pd.Series(dtype=float)

    return {
        "base_run_id": str(base_run["run_id"]),
        "candidate_run_id": str(candidate_run["run_id"]),
        "base_model_version": str(base_run["model_version"]),
        "candidate_model_version": str(candidate_run["model_version"]),
        "bakery_rows_compared": int(len(bakery_diff)),
        "sku_rows_compared": int(len(sku_diff)),
        "base_bakery_total": round(float(bakery_diff["base_forecast"].fillna(0.0).sum()), 6) if not bakery_diff.empty else 0.0,
        "candidate_bakery_total": round(float(bakery_diff["candidate_forecast"].fillna(0.0).sum()), 6) if not bakery_diff.empty else 0.0,
        "bakery_total_delta": round(float(bakery_diff["delta"].fillna(0.0).sum()), 6) if not bakery_diff.empty else 0.0,
        "mean_abs_bakery_delta": round(float(bakery_abs.mean()), 6) if not bakery_abs.empty else 0.0,
        "mean_abs_sku_delta": round(float(sku_abs.mean()), 6) if not sku_abs.empty else 0.0,
        "top_bakery_deltas": (
            bakery_diff.assign(abs_delta=bakery_diff["delta"].abs())
            .sort_values("abs_delta", ascending=False)
            .head(20)[["forecast_date", "bakery_id", "bakery_name", "base_forecast", "candidate_forecast", "delta"]]
            .to_dict("records")
            if not bakery_diff.empty
            else []
        ),
        "top_sku_deltas": (
            sku_diff.assign(abs_delta=sku_diff["delta"].abs())
            .sort_values("abs_delta", ascending=False)
            .head(20)[["forecast_date", "bakery_id", "product_id", "product_name", "base_forecast", "candidate_forecast", "delta"]]
            .to_dict("records")
            if not sku_diff.empty
            else []
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a candidate forecast run against the current active run")
    parser.add_argument("--candidate-run-id", required=True)
    parser.add_argument("--base-run-id", default=None, help="Optional explicit base run id. Defaults to current active run.")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH))
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    client = create_client(args.env_file)

    base_run_id = args.base_run_id or get_active_run_id(client)
    if not base_run_id:
        raise ValueError("No active run found and no --base-run-id provided")

    base_run = get_latest_run_row(client, base_run_id)
    candidate_run = get_latest_run_row(client, args.candidate_run_id)
    if not base_run:
        raise ValueError(f"Base run not found: {base_run_id}")
    if not candidate_run:
        raise ValueError(f"Candidate run not found: {args.candidate_run_id}")

    bakery_diff = compare_bakery_totals(client, base_run_id, args.candidate_run_id)
    sku_diff = compare_sku_totals(client, base_run_id, args.candidate_run_id)
    summary = build_summary(base_run, candidate_run, bakery_diff, sku_diff)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 72)
    print("RUN COMPARISON")
    print("=" * 72)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
