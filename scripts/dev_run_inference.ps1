param(
    [string]$EnvFile = ".env.dev",
    [string]$Scenario = "",
    [string]$ActivateRun = "",
    [string]$RunPrefix = "",
    [int]$HorizonDays = 0,
    [switch]$RefreshDatasets
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

if (-not (Test-Path -LiteralPath $EnvFile)) {
    throw "Dev env file not found: $EnvFile. Copy deploy\vm\forecast.dev.env.example to .env.dev and fill credentials."
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

$envMap = Read-EnvFile $EnvFile
foreach ($key in $envMap.Keys) {
    Set-Item -Path "env:$key" -Value $envMap[$key]
}

if (-not $env:CLICKHOUSE_DATABASE) {
    throw "CLICKHOUSE_DATABASE is required in $EnvFile"
}
if ($env:APP_ENV -eq "prod") {
    throw "Refusing to run dev inference with APP_ENV=prod in $EnvFile"
}
if (-not $env:FORECAST_TABLE_SUFFIX) {
    throw "FORECAST_TABLE_SUFFIX is required in $EnvFile for dev inference. Use a suffix like _dev."
}
if ($env:FORECAST_TABLE_SUFFIX -notmatch "^_[A-Za-z0-9_]+$") {
    throw "Invalid FORECAST_TABLE_SUFFIX='$($env:FORECAST_TABLE_SUFFIX)'. Use a suffix like _dev."
}

$python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}

$scenarioValue = if ($Scenario) { $Scenario } elseif ($env:FORECAST_SCENARIO) { $env:FORECAST_SCENARIO } else { "uplifted_norm" }
$activateValue = if ($ActivateRun) { $ActivateRun } elseif ($env:FORECAST_ACTIVATE_RUN) { $env:FORECAST_ACTIVATE_RUN } else { "none" }
$prefixValue = if ($RunPrefix) { $RunPrefix } elseif ($env:FORECAST_RUN_PREFIX) { $env:FORECAST_RUN_PREFIX } else { "dev" }
$horizonValue = if ($HorizonDays -gt 0) { $HorizonDays } elseif ($env:FORECAST_HORIZON_DAYS) { [int]$env:FORECAST_HORIZON_DAYS } else { 14 }

$args = @(
    "-m", "pipelines.forecast_publish.run_production_inference",
    "--env-file", $EnvFile,
    "--scenario", $scenarioValue,
    "--horizon-days", "$horizonValue",
    "--run-prefix", $prefixValue,
    "--activate-run", $activateValue,
    "--summary-path", "reports\dev_production_inference_summary.json",
    "--require-nonprod-tables"
)

if ($env:FORECAST_UPLIFT_PROFILE_VERSION) {
    $args += @("--uplift-profile-version", $env:FORECAST_UPLIFT_PROFILE_VERSION)
}
if ($RefreshDatasets -or $env:FORECAST_REFRESH_DATASETS -match "^(1|true|yes|on)$") {
    $args += "--refresh-datasets"
}

& $python @args
