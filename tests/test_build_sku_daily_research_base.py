import importlib.util
import json
from pathlib import Path
import shutil
import uuid

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "experiments_v2"
    / "build_sku_daily_research_base.py"
)
SPEC = importlib.util.spec_from_file_location("build_sku_daily_research_base", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_sku_daily_research_base_outputs_expected_files() -> None:
    tmp_path = Path.cwd() / ".pytest_local" / f"build_sku_daily_research_base_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    sales_path = tmp_path / "sales.csv"
    release_path = tmp_path / "release.csv"
    moves_path = tmp_path / "moves.csv"
    partner_path = tmp_path / "partner.csv"
    output_dir = tmp_path / "processed"
    audit_dir = tmp_path / "audit"

    sales_df = pd.DataFrame(
        [
            {
                "check_datetime": "2026-05-01 10:00:00",
                "check_date": "2026-05-01",
                "cash_event_type": "Продажа",
                "quantity": 2.0,
                "price": 100.0,
                "line_amount": 200.0,
                "bakery_id": "1",
                "bakery_name": "Bakery 1",
                "city": "Kazan",
                "product_id": "101",
                "product_name": "SKU A",
                "category_name": "Bread",
            },
            {
                "check_datetime": "2026-05-01 10:00:00",
                "check_date": "2026-05-01",
                "cash_event_type": "Продажа",
                "quantity": 2.0,
                "price": 100.0,
                "line_amount": 200.0,
                "bakery_id": "1",
                "bakery_name": "Bakery 1",
                "city": "Kazan",
                "product_id": "101",
                "product_name": "SKU A",
                "category_name": "Bread",
            },
            {
                "check_datetime": "2026-05-01 12:00:00",
                "check_date": "2026-05-01",
                "cash_event_type": "Продажа",
                "quantity": 1.0,
                "price": 120.0,
                "line_amount": 120.0,
                "bakery_id": "1",
                "bakery_name": "Bakery 1",
                "city": "Kazan",
                "product_id": "102",
                "product_name": "SKU B",
                "category_name": "Bread",
            },
        ]
    )
    sales_df.to_csv(sales_path, index=False, encoding="utf-8-sig")

    release_df = pd.DataFrame(
        [
            {
                "_UUID": "u1",
                "release_id": "r1",
                "line_id": "l1",
                "release_date": "2026-05-01",
                "bakery_id": "1",
                "product_id": "101",
                "quantity": 5.0,
                "baker_name": "Ivan",
            },
            {
                "_UUID": "u1",
                "release_id": "r1",
                "line_id": "l1",
                "release_date": "2026-05-01",
                "bakery_id": "1",
                "product_id": "101",
                "quantity": 5.0,
                "baker_name": "Ivan",
            },
            {
                "_UUID": "u2",
                "release_id": "r1",
                "line_id": "l1",
                "release_date": "2026-05-01",
                "bakery_id": "1",
                "product_id": "101",
                "quantity": 6.0,
                "baker_name": "Petr",
            },
        ]
    )
    release_df.to_csv(release_path, index=False, encoding="utf-8-sig")

    moves_df = pd.DataFrame(
        [
            {
                "move_id": "m1",
                "move_date": "2026-05-01",
                "product_id": "101",
                "sender_id": "2",
                "receiver_id": "1",
                "quantity": 3.0,
            },
            {
                "move_id": "m2",
                "move_date": "2026-05-01",
                "product_id": "101",
                "sender_id": "1",
                "receiver_id": "3",
                "quantity": 1.0,
            },
        ]
    )
    moves_df.to_csv(moves_path, index=False, encoding="utf-8-sig")

    partner_df = pd.DataFrame(
        [
            {
                "kkt_id": "k1",
                "kkt_name": "KKT 1",
                "kkt_number": "111",
                "organization_id": "org1",
                "organization_name": "Org One",
                "bakery_id": "1",
            },
            {
                "kkt_id": "k2",
                "kkt_name": "KKT 2",
                "kkt_number": "222",
                "organization_id": "org2",
                "organization_name": "Org Two",
                "bakery_id": "1",
            },
        ]
    )
    partner_df.to_csv(partner_path, index=False, encoding="utf-8-sig")

    try:
        paths = MODULE.build_sku_daily_research_base(
            sales_path=sales_path,
            release_path=release_path,
            moves_path=moves_path,
            partner_path=partner_path,
            output_dir=output_dir,
            audit_dir=audit_dir,
            chunk_size=10,
            panel_min_observed_days=1,
        )

        dataset = pd.read_csv(paths["dataset"], encoding="utf-8-sig")
        panel = pd.read_csv(paths["panel"], encoding="utf-8-sig")
        assert len(dataset) == 2
        assert len(panel) == 2

        sku_a = dataset.loc[dataset["product_id"] == 101].iloc[0]
        assert sku_a["observed_sales_qty"] == 2.0
        assert sku_a["release_qty"] == 11.0
        assert sku_a["incoming_move_qty"] == 3.0
        assert sku_a["outgoing_move_qty"] == 1.0
        assert sku_a["net_move_qty"] == 2.0
        assert sku_a["organization_conflict_flag"] == 1
        assert sku_a["release_conflict_flag"] == 1

        summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
        assert summary["rows"] == 2
        assert summary["sales_dedup"]["removed_rows"] == 1
        assert summary["release_dedup"]["conflict_groups"] == 1
        assert summary["partner_map"]["partner_conflict_bakeries"] == 1

        release_conflicts = pd.read_csv(audit_dir / "release_conflicts.csv", encoding="utf-8-sig")
        partner_conflicts = pd.read_csv(audit_dir / "partner_conflicts.csv", encoding="utf-8-sig")
        assert len(release_conflicts) == 2
        assert len(partner_conflicts) == 2
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_build_full_panel_adds_missing_zero_days() -> None:
    df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-05-01"),
                "bakery_id": "b1",
                "bakery_name": "Bakery 1",
                "city": "Kazan",
                "product_id": "p1",
                "product_name": "SKU A",
                "category_name": "Bread",
                "observed_sales_qty": 2.0,
                "observed_sales_amount": 200.0,
                "sales_rows_count": 1,
                "sales_hours_count": 1,
                "price_x_qty": 200.0,
                "priced_qty": 2.0,
                "sales_present_flag": 1,
                "sales_dedup_applied_flag": 1,
                "release_qty": 0.0,
                "release_rows_count": 0,
                "release_bakers_count": 0,
                "release_has_data_flag": 0,
                "release_present_flag": 0,
                "release_conflict_flag": 0,
                "incoming_move_qty": 0.0,
                "incoming_move_rows_count": 0,
                "outgoing_move_qty": 0.0,
                "outgoing_move_rows_count": 0,
                "net_move_qty": 0.0,
                "has_incoming_move_flag": 0,
                "has_outgoing_move_flag": 0,
                "moves_present_flag": 0,
                "moves_conflict_flag": 0,
                "organization_conflict_flag": 0,
            },
            {
                "date": pd.Timestamp("2026-05-03"),
                "bakery_id": "b1",
                "bakery_name": "Bakery 1",
                "city": "Kazan",
                "product_id": "p1",
                "product_name": "SKU A",
                "category_name": "Bread",
                "observed_sales_qty": 3.0,
                "observed_sales_amount": 300.0,
                "sales_rows_count": 1,
                "sales_hours_count": 1,
                "price_x_qty": 300.0,
                "priced_qty": 3.0,
                "sales_present_flag": 1,
                "sales_dedup_applied_flag": 1,
                "release_qty": 0.0,
                "release_rows_count": 0,
                "release_bakers_count": 0,
                "release_has_data_flag": 0,
                "release_present_flag": 0,
                "release_conflict_flag": 0,
                "incoming_move_qty": 0.0,
                "incoming_move_rows_count": 0,
                "outgoing_move_qty": 0.0,
                "outgoing_move_rows_count": 0,
                "net_move_qty": 0.0,
                "has_incoming_move_flag": 0,
                "has_outgoing_move_flag": 0,
                "moves_present_flag": 0,
                "moves_conflict_flag": 0,
                "organization_conflict_flag": 0,
            },
        ]
    )

    panel = MODULE.build_full_panel(df, min_observed_days=1)
    assert len(panel) == 3
    middle_day = panel.loc[pd.to_datetime(panel["date"]) == pd.Timestamp("2026-05-02")].iloc[0]
    assert middle_day["observed_sales_qty"] == 0.0
    assert middle_day["sales_present_flag"] == 0
