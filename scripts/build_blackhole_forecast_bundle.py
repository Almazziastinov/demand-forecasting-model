from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "blackhole_forecast_bundle"
DEFAULT_PROFILE_CHUNK_MB = 35


BUNDLE_FILES = [
    Path("pipelines/forecast_publish/nightly_refresh.py"),
    Path("pipelines/forecast_publish/load_forecast_run.py"),
    Path("pipelines/forecast_publish/activate_run.py"),
    Path("pipelines/forecast_publish/sku_hour_profile_store.py"),
    Path("scripts/export_clickhouse_checks.py"),
    Path("scripts/clickhouse_export_template.sql"),
    Path("src/config.py"),
    Path("src/experiments_v2/__init__.py"),
    Path("src/experiments_v2/common.py"),
    Path("src/experiments_v2/apply_bakery_profiles.py"),
    Path("src/experiments_v2/bakery_day_forecast.py"),
    Path("src/experiments_v2/build_bakery_daily_dataset.py"),
    Path("src/experiments_v2/build_bakery_hour_profile.py"),
    Path("src/experiments_v2/raw_sales_dedup.py"),
    Path("src/experiments_v2/raw_snapshot_schema.py"),
]

SERVER_REQUIREMENTS = """\
clickhouse-connect==1.0.0
joblib==1.5.3
lightgbm==4.6.0
numpy==2.4.5
pandas==3.0.1
scikit-learn==1.8.0
"""

SERVER_ENV_TEMPLATE = """\
HOST={host}
PORT={port}
USER={user}
PASSWORD={password}
DATABASE={database}
"""

SERVICE_UNIT = """\
[Unit]
Description=Bakery forecast nightly refresh
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/opt/forecast_job
EnvironmentFile=/opt/forecast_job/.env
ExecStart=/opt/forecast_job/.venv/bin/python /opt/forecast_job/pipelines/forecast_publish/nightly_refresh.py --profile-source clickhouse
"""

TIMER_UNIT = """\
[Unit]
Description=Run bakery forecast nightly at midnight Moscow time

[Timer]
OnCalendar=*-*-* 00:00:00 Europe/Moscow
Persistent=true
Unit=bakery-forecast-nightly.service

[Install]
WantedBy=timers.target
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def stage_bundle(output_dir: Path) -> Path:
    stage_dir = output_dir / "bundle_root"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    for relative_path in BUNDLE_FILES:
        source = ROOT / relative_path
        target = stage_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    write_text(stage_dir / "requirements-forecast.txt", SERVER_REQUIREMENTS)
    write_text(stage_dir / ".env.example", SERVER_ENV_TEMPLATE.format(
        host="rc1b-aergg94cc1r6ctr1.mdb.yandexcloud.net",
        port="8443",
        user="your_clickhouse_user",
        password="your_clickhouse_password",
        database="Svezhar",
    ))
    write_text(stage_dir / "deploy" / "bakery-forecast-nightly.service", SERVICE_UNIT)
    write_text(stage_dir / "deploy" / "bakery-forecast-nightly.timer", TIMER_UNIT)
    return stage_dir


def build_tarball(stage_dir: Path, output_dir: Path) -> Path:
    tarball_path = output_dir / "forecast_job_bundle.tar.gz"
    if tarball_path.exists():
        tarball_path.unlink()
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(stage_dir, arcname=".")
    return tarball_path


def gzip_profile(profile_path: Path, output_dir: Path) -> Path:
    gz_path = output_dir / f"{profile_path.name}.gz"
    with profile_path.open("rb") as source, gzip.open(gz_path, "wb", compresslevel=6) as target:
        shutil.copyfileobj(source, target)
    return gz_path


def split_file(path: Path, output_dir: Path, chunk_mb: int) -> list[Path]:
    chunk_size = chunk_mb * 1024 * 1024
    chunk_paths: list[Path] = []
    with path.open("rb") as handle:
        index = 1
        while True:
            data = handle.read(chunk_size)
            if not data:
                break
            chunk_path = output_dir / f"{path.name}.part{index:03d}"
            chunk_path.write_bytes(data)
            chunk_paths.append(chunk_path)
            index += 1
    return chunk_paths


def build_manifest(
    *,
    tarball_path: Path,
    profile_gz_path: Path,
    chunk_paths: list[Path],
    output_dir: Path,
) -> Path:
    manifest = {
        "bundle": {
            "path": str(tarball_path),
            "size_bytes": tarball_path.stat().st_size,
            "sha256": sha256_file(tarball_path),
        },
        "profile_gzip": {
            "path": str(profile_gz_path),
            "size_bytes": profile_gz_path.stat().st_size,
            "sha256": sha256_file(profile_gz_path),
        },
        "profile_chunks": [
            {
                "path": str(chunk_path),
                "size_bytes": chunk_path.stat().st_size,
                "sha256": sha256_file(chunk_path),
            }
            for chunk_path in chunk_paths
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Blackhole server bundle for nightly forecast job")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--profile-path",
        default=str(ROOT / "data" / "processed" / "sku_hour_share_profile_smoothed.csv"),
    )
    parser.add_argument("--profile-chunk-mb", type=int, default=DEFAULT_PROFILE_CHUNK_MB)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_dir = stage_bundle(output_dir)
    tarball_path = build_tarball(stage_dir, output_dir)
    profile_gz_path = gzip_profile(Path(args.profile_path), output_dir)
    chunk_paths = split_file(profile_gz_path, output_dir, args.profile_chunk_mb)
    manifest_path = build_manifest(
        tarball_path=tarball_path,
        profile_gz_path=profile_gz_path,
        chunk_paths=chunk_paths,
        output_dir=output_dir,
    )

    print("=" * 72)
    print("BLACKHOLE FORECAST BUNDLE READY")
    print("=" * 72)
    print(f"bundle: {tarball_path}")
    print(f"profile_gzip: {profile_gz_path}")
    print(f"profile_chunks: {len(chunk_paths)}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
