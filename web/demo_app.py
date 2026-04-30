"""
Artifact-based demo app for the monthly hybrid benchmark snapshot.

This app does not run training or inference.
It visualizes saved benchmark artifacts from:
- Full_benchmark_mounth_results/full_benchmark_monthly
- Full_benchmark_mounth_results/sku_local_monthly
"""

from __future__ import annotations

import os
import sys

import pandas as pd
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.demo_artifact_store import bakery_comparison_table, build_series, filter_products, load_store  # noqa: E402
from web.demo_artifact_store import resolve_compare_model  # noqa: E402


app = Flask(__name__)


def _default_models(models: list[str]) -> tuple[str, str]:
    real_models = [model for model in models if model != "avg_lag_7_14_core"]
    if "66_cluster_features" in real_models:
        main_model = "66_cluster_features"
    elif real_models:
        main_model = real_models[0]
    else:
        main_model = models[0] if models else ""
    if main_model != "60_baseline_v3" and "60_baseline_v3" in real_models:
        compare_model = "60_baseline_v3"
    else:
        compare_model = real_models[1] if len(real_models) > 1 else (real_models[0] if real_models else main_model)
    return main_model, compare_model


@app.route("/")
def index():
    store = load_store()
    models = store["models"]
    bakeries = store["bakeries"]
    buckets = store["buckets"]

    default_bakery = bakeries[0] if bakeries else ""
    default_bucket = "all"
    default_products = filter_products(default_bakery, default_bucket) if default_bakery else pd.DataFrame()
    default_product = default_products.iloc[0]["product"] if not default_products.empty else ""
    default_main_model, default_compare_model = _default_models(models)

    return render_template(
        "demo_dashboard.html",
        bakeries=bakeries,
        buckets=buckets,
        models=models,
        default_bakery=default_bakery,
        default_bucket=default_bucket,
        default_product=default_product,
        default_main_model=default_main_model,
        default_compare_model=default_compare_model,
        date_min=store["date_min"],
        date_max=store["date_max"],
        date_range_label=store["date_range_label"],
    )


@app.route("/api/products")
def api_products():
    bakery = request.args.get("bakery", "")
    bucket = request.args.get("bucket", "all")
    products = filter_products(bakery, bucket) if bakery else pd.DataFrame()
    if products.empty:
        return jsonify({"products": [], "best_models": [], "best_r2": []})

    return jsonify(
        {
            "products": products[["product", "best_model", "best_r2", "r2_bucket"]].to_dict("records"),
            "best_models": products["best_model"].fillna("").tolist(),
            "best_r2": [None if pd.isna(v) else float(v) for v in products["best_r2"].tolist()],
        }
    )


@app.route("/api/preview")
def api_preview():
    bakery = request.args.get("bakery", "")
    product = request.args.get("product", "")
    main_model = request.args.get("main_model", "66_cluster_features")
    compare_model = request.args.get("compare_model", "60_baseline_v3")
    bucket = request.args.get("bucket", "all")

    if not bakery:
        return jsonify({"error": "bakery is required"}), 400

    products = filter_products(bakery, bucket)
    if products.empty:
        return jsonify({"error": "No products for selected bakery/bucket"}), 404

    product_names = products["product"].tolist()
    if not product or product not in product_names:
        product = product_names[0]

    resolved_compare_model = resolve_compare_model(main_model, compare_model)
    payload = build_series(bakery, product, main_model, resolved_compare_model)
    table_payload = bakery_comparison_table(bakery, main_model, resolved_compare_model, bucket)

    return jsonify(
        {
            "bakery": bakery,
            "product": product,
            "bucket": bucket,
            "main_model": main_model,
            "compare_model": resolved_compare_model,
            "series": payload["series"],
            "metrics": payload["metrics"],
            "sku_info": payload["sku_info"],
            "bakery_table": table_payload,
            "products": products[["product", "best_model", "best_r2", "r2_bucket"]].to_dict("records"),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
