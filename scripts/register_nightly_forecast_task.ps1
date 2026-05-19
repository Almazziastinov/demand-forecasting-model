param(
    [string]$TaskName = "DemandForecastNightlyRefresh",
    [string]$ProjectRoot = "C:\Users\dns\Desktop\Projects\demand-forecasting-model",
    [string]$PythonExe = "C:\Users\dns\Desktop\Projects\demand-forecasting-model\.venv\Scripts\python.exe",
    [string]$RunAt = "00:00"
)

$scriptPath = Join-Path $ProjectRoot "pipelines\forecast_publish\nightly_refresh.py"
$actionArgs = "`"$scriptPath`""

$action = New-ScheduledTaskAction -Execute $PythonExe -Argument $actionArgs -WorkingDirectory $ProjectRoot
$trigger = New-ScheduledTaskTrigger -Daily -At $RunAt
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 12) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 30)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Nightly bakery forecast refresh, publish, and activation"
