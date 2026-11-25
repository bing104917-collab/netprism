param(
    [Parameter(Mandatory = $true)]
    [string]$Exe,
    [int]$Port = 8791,
    [int]$TimeoutSec = 120
)

$ErrorActionPreference = "Stop"
$project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$exePath = (Resolve-Path $Exe).Path
$logDir = Join-Path $env:TEMP "netprism_exe_test"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stdout = Join-Path $logDir "stdout.log"
$stderr = Join-Path $logDir "stderr.log"

Write-Host "== smoke: $exePath --version ==" -ForegroundColor Cyan
& $exePath --version
if ($LASTEXITCODE -ne 0) { throw "exe --version failed" }

$oldNoBrowser = $env:NETPRISM_NO_BROWSER
$oldPort = $env:NETPRISM_PORT
$env:NETPRISM_NO_BROWSER = "1"
$env:NETPRISM_PORT = "$Port"
$proc = Start-Process -FilePath $exePath -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden
Write-Host "== started pid $($proc.Id), waiting for http://127.0.0.1:$Port/ ==" -ForegroundColor Cyan
try {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $health = $null
    while ((Get-Date) -lt $deadline) {
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 3
            break
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    if (-not $health -or -not $health.ok) {
        Write-Host "---- stdout ----" -ForegroundColor Yellow
        Get-Content $stdout -Tail 40 -ErrorAction SilentlyContinue
        Write-Host "---- stderr ----" -ForegroundColor Yellow
        Get-Content $stderr -Tail 40 -ErrorAction SilentlyContinue
        throw "server did not become healthy"
    }
    $index = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/index" -TimeoutSec 10
    Write-Host ("health: {0} {1}; models: {2}; first: {3}" -f $health.ok, $health.version, $index.models.Count, $index.models[0].name)
    $first = $index.models[0]
    $graph = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/model/$($first.id)" -TimeoutSec 10
    Write-Host ("graph: {0} mode={1} layers={2} nodes={3}" -f $graph.model.name, $graph.model.mode, $graph.model.numLayers, $graph.nodes.Count)
    $page = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/" -UseBasicParsing -TimeoutSec 10
    Write-Host ("page: HTTP {0}, bytes {1}" -f $page.StatusCode, $page.Content.Length)
    if ($index.models.Count -lt 8) { throw "expected 8 bundled demo models, got $($index.models.Count)" }
    Write-Host "EXE SMOKE OK" -ForegroundColor Green
} finally {
    if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
    $env:NETPRISM_NO_BROWSER = $oldNoBrowser
    $env:NETPRISM_PORT = $oldPort
}
