from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from openpyxl import Workbook

from src.pilot_plan_archive import (
    LEGACY_HEADERS,
    STOCK_AWARE_HEADERS,
    PilotPlanManifestRecord,
    build_manifest_record,
    normalize_header,
    parse_pilot_plan,
    validate_manifest,
)


def _write_workbook(
    path: Path, title: str, headers: tuple[str, ...], row: list[object]
) -> None:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append([title])
    worksheet.append(list(headers))
    worksheet.append(row)
    workbook.save(path)


def test_parse_legacy_v1(tmp_path: Path) -> None:
    path = tmp_path / "plan_23.07.2026.xlsx"
    _write_workbook(
        path,
        "Прогноз выпечки — 23.07.2026 (Четверг)",
        LEGACY_HEADERS,
        ["20", "Выпечка сладкая", "Ватрушка", 26.1, 30, 10],
    )

    plan = parse_pilot_plan(path)

    assert plan.target_date == date(2026, 7, 23)
    assert plan.format_version == "legacy_v1"
    assert len(plan.sha256) == 64
    assert plan.rows[0].issued_forecast == 26.1
    assert plan.rows[0].plan_qty == 30
    assert plan.rows[0].yesterday_stock is None


def test_parse_stock_aware_v2_with_normalized_headers(tmp_path: Path) -> None:
    path = tmp_path / "publication.xlsx"
    headers = tuple(f"  {header.upper()}  " for header in STOCK_AWARE_HEADERS)
    _write_workbook(
        path,
        "Прогноз выпечки — 10.08.2026 (Понедельник)",
        headers,
        ["Мира 45", "Выпечка сладкая", "Ватрушка", 2.6, 1, 1.6, 10, 11, 10],
    )

    plan = parse_pilot_plan(path)
    row = plan.rows[0]

    assert plan.format_version == "stock_aware_v2"
    assert row.yesterday_stock == 1
    assert row.net_need == 1.6
    assert row.plan_qty == 10
    assert row.total_for_sale == 11


def test_target_date_falls_back_to_filename(tmp_path: Path) -> None:
    path = tmp_path / "plan_31.07.2026.xlsx"
    _write_workbook(
        path,
        "Ежедневный план",
        LEGACY_HEADERS,
        ["20", "Выпечка сладкая", "Ватрушка", 1, 10, 10],
    )

    assert parse_pilot_plan(path).target_date == date(2026, 7, 31)


def test_normalize_header_handles_whitespace_case_and_yo() -> None:
    assert normalize_header("  ВСЁ   НА   ПРОДАЖУ ") == "все на продажу"


def test_build_manifest_preserves_provenance(tmp_path: Path) -> None:
    path = tmp_path / "plan_10.08.2026.xlsx"
    _write_workbook(
        path,
        "Plan",
        LEGACY_HEADERS,
        ["20", "Выпечка сладкая", "Ватрушка", 1, 10, 10],
    )
    plan = parse_pilot_plan(path)

    record = build_manifest_record(
        plan,
        source_quality="bitrix_attachment",
        published_at=datetime(2026, 8, 10, 6, tzinfo=timezone.utc),
        message_id="123",
        disk_id="456",
        is_effective=True,
    )

    assert record.rows == 1
    assert record.message_id == "123"
    assert record.disk_id == "456"
    assert record.published_at == datetime(2026, 8, 10, 6, tzinfo=timezone.utc)
    assert record.is_effective


def test_bitrix_source_requires_message_id(tmp_path: Path) -> None:
    path = tmp_path / "plan_10.08.2026.xlsx"
    _write_workbook(
        path,
        "Plan",
        LEGACY_HEADERS,
        ["20", "Выпечка сладкая", "Ватрушка", 1, 10, 10],
    )

    with pytest.raises(ValueError, match="require message_id"):
        build_manifest_record(
            parse_pilot_plan(path), source_quality="bitrix_attachment"
        )


def _record(*, sha256: str, effective: bool = False) -> PilotPlanManifestRecord:
    return PilotPlanManifestRecord(
        target_date=date(2026, 8, 10),
        format_version="stock_aware_v2",
        rows=10,
        sha256=sha256,
        source_quality="local_unverified",
        path=f"{sha256}.xlsx",
        is_effective=effective,
    )


def test_manifest_rejects_conflicting_unselected_versions() -> None:
    with pytest.raises(ValueError, match="Conflicting unselected"):
        validate_manifest([_record(sha256="a"), _record(sha256="b")])


def test_manifest_accepts_explicit_effective_choice() -> None:
    validate_manifest([_record(sha256="a", effective=True), _record(sha256="b")])


def test_manifest_rejects_multiple_effective_versions() -> None:
    with pytest.raises(ValueError, match="Multiple effective"):
        validate_manifest(
            [_record(sha256="a", effective=True), _record(sha256="b", effective=True)]
        )


def test_manifest_rejects_identity_with_conflicting_hashes() -> None:
    first = _record(sha256="a")
    second = _record(sha256="b")
    first = PilotPlanManifestRecord(**{**first.__dict__, "message_id": "1"})
    second = PilotPlanManifestRecord(**{**second.__dict__, "message_id": "1"})

    with pytest.raises(ValueError, match="conflicting hashes"):
        validate_manifest([first, second])
