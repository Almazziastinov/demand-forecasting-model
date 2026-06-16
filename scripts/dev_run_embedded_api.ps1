param(
    [string]$EnvFile = ".env.dev",
    [int]$Port = 0
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location (Join-Path $Root "apps\forecast_embedded")

if (-not (Test-Path -LiteralPath (Join-Path $Root $EnvFile))) {
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

$envPath = Join-Path $Root $EnvFile
$envMap = Read-EnvFile $envPath
foreach ($key in $envMap.Keys) {
    Set-Item -Path "env:$key" -Value $envMap[$key]
}

$env:ENV_FILE = $envPath
if (-not $env:APP_ENV) {
    $env:APP_ENV = "dev"
}
if ($Port -gt 0) {
    $env:PORT = "$Port"
}
if ($env:APP_ENV -eq "prod") {
    throw "Refusing to run dev embedded API with APP_ENV=prod in $EnvFile"
}
if (-not $env:FORECAST_TABLE_SUFFIX) {
    throw "FORECAST_TABLE_SUFFIX is required in $EnvFile for dev embedded API. Use a suffix like _dev."
}
if ($env:FORECAST_TABLE_SUFFIX -notmatch "^_[A-Za-z0-9_]+$") {
    throw "Invalid FORECAST_TABLE_SUFFIX='$($env:FORECAST_TABLE_SUFFIX)'. Use a suffix like _dev."
}

$python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}

$portValue = if ($env:PORT) { $env:PORT } else { "3001" }
& $python -m uvicorn app.main:app --host 127.0.0.1 --port $portValue --reload
