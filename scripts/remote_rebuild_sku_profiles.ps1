param(
    [string]$VmEnvFile = ".codex\prod_vm.env",
    [string]$ClickhouseEnvFile = ".env.dev",
    [string]$SshKeyPath = "",
    [string]$RemoteRoot = "/root/demand-forecasting-model",
    [string]$DateFrom = "2026-01-01",
    [string]$DateTo = "",
    [string]$ProfileVersion = "dev_assortment_20260618",
    [string]$ProfileTable = "sku_hour_share_profile_smoothed_embedded_dev",
    [string]$UpliftTable = "sku_hour_uplift_multiplier_embedded_dev",
    [switch]$DisableAssortmentFilter,
    [switch]$SkipExport,
    [switch]$SkipInstall,
    [switch]$Background
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

if (-not $DateTo) {
    $DateTo = (Get-Date).AddDays(-1).ToString("yyyy-MM-dd")
}

function Read-EnvFile([string]$Path) {
    $result = @{}
    Get-Content -LiteralPath $Path | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
            return
        }
        $key, $value = $line.Split("=", 2)
        $result[$key.Trim()] = $value.Trim().Trim('"').Trim("'")
    }
    return $result
}

if (-not (Test-Path -LiteralPath $VmEnvFile)) {
    throw "VM env file not found: $VmEnvFile"
}
if (-not (Test-Path -LiteralPath $ClickhouseEnvFile)) {
    throw "ClickHouse env file not found: $ClickhouseEnvFile"
}

$vm = Read-EnvFile $VmEnvFile
$hostName = if ($vm["NEW_PROD_VM_HOST"]) { $vm["NEW_PROD_VM_HOST"] } else { $vm["PROD_VM_HOST"] }
$userName = if ($vm["NEW_PROD_VM_USER"]) { $vm["NEW_PROD_VM_USER"] } else { $vm["PROD_VM_USER"] }
if (-not $hostName -or -not $userName) {
    throw "NEW_PROD_VM_HOST/NEW_PROD_VM_USER or PROD_VM_HOST/PROD_VM_USER are required in $VmEnvFile"
}

if (-not $SshKeyPath) {
    $SshKeyPath = if ($vm["NEW_PROD_VM_KEY"]) { $vm["NEW_PROD_VM_KEY"] } else { $vm["PROD_VM_KEY"] }
}
if (-not $SshKeyPath -or -not (Test-Path -LiteralPath $SshKeyPath)) {
    throw "SSH key is required. Pass -SshKeyPath or set NEW_PROD_VM_KEY/PROD_VM_KEY in $VmEnvFile."
}

$sshExe = Join-Path $env:WINDIR "System32\OpenSSH\ssh.exe"
$scpExe = Join-Path $env:WINDIR "System32\OpenSSH\scp.exe"
if (-not (Test-Path -LiteralPath $sshExe)) {
    $sshExe = "ssh"
}
if (-not (Test-Path -LiteralPath $scpExe)) {
    $scpExe = "scp"
}

$target = "$userName@$hostName"
$sshBase = @(
    "-i", $SshKeyPath,
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=20",
    $target
)

function Invoke-Remote([string]$Command) {
    & $sshExe @sshBase $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Remote command failed with exit code $LASTEXITCODE"
    }
}

function Copy-ToRemote([string]$LocalPath, [string]$RemotePath) {
    & $scpExe -i $SshKeyPath -o StrictHostKeyChecking=accept-new $LocalPath "$target`:$RemotePath"
    if ($LASTEXITCODE -ne 0) {
        throw "scp failed with exit code $LASTEXITCODE"
    }
}

$tmpDir = Join-Path ([System.IO.Path]::GetTempPath()) ("dfm_remote_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tmpDir | Out-Null
$stageDir = Join-Path $tmpDir "repo"
$archivePath = Join-Path $tmpDir "repo.tar.gz"
$remoteArchive = "/tmp/demand_forecasting_model_repo.tar.gz"
$remoteRunner = "/tmp/rebuild_sku_profiles.sh"

$packageDirs = @(
    "apps",
    "config",
    "pipelines",
    "scripts",
    "src",
    "tests"
) | Where-Object { Test-Path -LiteralPath $_ }
$packageFiles = @(
    "requirements.txt",
    "requirements-dev.txt"
) | Where-Object { Test-Path -LiteralPath $_ }

New-Item -ItemType Directory -Path $stageDir | Out-Null
foreach ($dir in $packageDirs) {
    Get-ChildItem -LiteralPath $dir -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object {
            $_.FullName -notmatch '\\__pycache__\\' -and
            $_.FullName -notmatch '\\.pytest_cache\\' -and
            $_.Extension -in @(".py", ".sql", ".ps1", ".txt", ".md", ".json")
        } |
        ForEach-Object {
            $relative = Resolve-Path -LiteralPath $_.FullName -Relative
            $relative = $relative.TrimStart(".\")
            $targetPath = Join-Path $stageDir $relative
            New-Item -ItemType Directory -Force -Path (Split-Path $targetPath) | Out-Null
            Copy-Item -LiteralPath $_.FullName -Destination $targetPath -Force
        }
}
foreach ($file in $packageFiles) {
    Copy-Item -LiteralPath $file -Destination (Join-Path $stageDir $file) -Force
}

tar -czf $archivePath -C $stageDir .
if ($LASTEXITCODE -ne 0) {
    throw "tar failed with exit code $LASTEXITCODE"
}

Invoke-Remote "mkdir -p $RemoteRoot/reports/required_assortment $RemoteRoot/data/raw $RemoteRoot/data/processed"
Copy-ToRemote $archivePath $remoteArchive
Copy-ToRemote $ClickhouseEnvFile "$RemoteRoot/.env.dev"
Copy-ToRemote "reports\required_assortment\assortment_city_products.csv" "$RemoteRoot/reports/required_assortment/assortment_city_products.csv"
Copy-ToRemote "reports\required_assortment\dim_products_lookup.csv" "$RemoteRoot/reports/required_assortment/dim_products_lookup.csv"

$skipExportValue = if ($SkipExport) { "1" } else { "0" }
$skipInstallValue = if ($SkipInstall) { "1" } else { "0" }
$disableAssortmentFilterValue = if ($DisableAssortmentFilter) { "1" } else { "0" }
$remoteScript = @"
set -euo pipefail

REMOTE_ROOT="$RemoteRoot"
ARCHIVE="$remoteArchive"
DATE_FROM="$DateFrom"
DATE_TO="$DateTo"
PROFILE_VERSION="$ProfileVersion"
PROFILE_TABLE="$ProfileTable"
UPLIFT_TABLE="$UpliftTable"
SKIP_EXPORT="$skipExportValue"
SKIP_INSTALL="$skipInstallValue"
DISABLE_ASSORTMENT_FILTER="$disableAssortmentFilterValue"

mkdir -p "`$REMOTE_ROOT"
cd "`$REMOTE_ROOT"
tar -xzf "`$ARCHIVE" -C "`$REMOTE_ROOT"

if [ -d .venv ] && [ ! -f .venv/bin/activate ]; then
  rm -rf .venv
fi

if [ ! -d .venv ]; then
  if ! python3 -m venv --help >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-pip
  fi
  python3 -m venv .venv
fi
. .venv/bin/activate

if [ "`$SKIP_INSTALL" != "1" ]; then
  python -m pip install --upgrade pip
  pip install pandas==3.0.1 numpy==2.4.4 clickhouse-connect joblib lightgbm scikit-learn
fi

mkdir -p data/raw data/processed reports/required_assortment

if [ "`$SKIP_EXPORT" != "1" ] && [ ! -s data/raw/sales_hrs_all_clickhouse.csv ]; then
  python scripts/export_clickhouse_checks.py \
    --env-file .env.dev \
    --sql-template scripts/clickhouse_export_template.sql \
    --date-from "`$DATE_FROM" \
    --date-to "`$DATE_TO" \
    --batch-mode weekly \
    --output data/raw/sales_hrs_all_clickhouse.csv
elif [ -s data/raw/sales_hrs_all_clickhouse.csv ]; then
  echo "Reusing existing data/raw/sales_hrs_all_clickhouse.csv"
fi

PROFILE_ARGS=(
  --source-path data/raw/sales_hrs_all_clickhouse.csv
  --output-dir data/processed
  --product-lookup-path reports/required_assortment/dim_products_lookup.csv
)
if [ "`$DISABLE_ASSORTMENT_FILTER" != "1" ]; then
  PROFILE_ARGS+=(--assortment-path reports/required_assortment/assortment_city_products.csv)
fi
python -m src.experiments_v2.build_sku_hour_share_profile "`${PROFILE_ARGS[@]}"

python -m src.experiments_v2.smooth_sku_hour_share_profile \
  --profile-path data/processed/sku_hour_share_profile.csv \
  --applied-path data/processed/sku_hour_share_profile_daily.csv \
  --output-dir data/processed

python -m pipelines.forecast_publish.sku_hour_profile_store \
  --mode load \
  --env-file .env.dev \
  --table "`$PROFILE_TABLE" \
  --profile-path data/processed/sku_hour_share_profile_smoothed.csv \
  --truncate

python -m pipelines.forecast_publish.sku_hour_profile_store \
  --mode load-uplift-multipliers \
  --env-file .env.dev \
  --uplift-table "`$UPLIFT_TABLE" \
  --applied-path data/processed/sku_hour_share_profile_daily_smoothed.csv \
  --profile-version "`$PROFILE_VERSION" \
  --truncate

python - <<'PY'
from pathlib import Path
import json
for path in [
    Path("data/processed/sku_hour_share_profile_summary.json"),
    Path("data/processed/sku_hour_share_profile_smoothed_summary.json"),
]:
    print(path)
    if path.exists():
        print(path.read_text(encoding="utf-8"))
PY
"@

$runnerPath = Join-Path $tmpDir "rebuild_sku_profiles.sh"
[System.IO.File]::WriteAllText(
    $runnerPath,
    $remoteScript,
    [System.Text.UTF8Encoding]::new($false)
)
Copy-ToRemote $runnerPath $remoteRunner
if ($Background) {
    $remoteLog = "$RemoteRoot/logs/rebuild_sku_profiles_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
    Invoke-Remote "mkdir -p $RemoteRoot/logs; setsid -f bash $remoteRunner </dev/null > $remoteLog 2>&1; echo $remoteLog"
} else {
    Invoke-Remote "bash $remoteRunner"
}

Remove-Item -LiteralPath $tmpDir -Recurse -Force
