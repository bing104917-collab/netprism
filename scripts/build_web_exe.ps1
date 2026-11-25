# Build the lean NetPrism Web EXE: double-click -> local server -> browser.
#
# Usage (from the repo root):
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_web_exe.ps1
#   powershell ... -File scripts\build_web_exe.ps1 -Python "C:\Python312\python.exe"
#
# Output: dist\web\NetPrism.exe  (single file, console window, ~30 MB)
#
# Deliberately excludes torch / onnx / PySide6: the browser app only needs the
# static AST parser, the bundled demo data and the web/ frontend.  A heavy
# build (trace + ONNX + TorchScript) is still available through
# scripts\build_exe.ps1 -Mode onefile.

param(
    [string]$Python = "",
    [switch]$SkipSmoke
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

if (-not $Python) {
    $Python = $env:NETPRISM_PYTHON
}
if (-not $Python) {
    $Python = "C:\Users\hrbkcy2523\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
}
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$Entry = Join-Path $Root "packaging\web_launcher.py"
$Icon = Join-Path $Root "packaging\netprism.ico"
$Dist = Join-Path $Root "dist\web"
$Work = Join-Path $Root "build\web"

if (-not (Test-Path $Entry)) { throw "Missing entry point: $Entry" }
if (-not (Test-Path (Join-Path $Root "web\index.html"))) { throw "Missing web/ frontend" }
if (-not (Test-Path (Join-Path $Root "examples\demo\index.json"))) { throw "Missing examples/demo data" }

& $Python -c "import PyInstaller, sys; print('PyInstaller', PyInstaller.__version__)"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller is not installed for $Python" }

New-Item -ItemType Directory -Force -Path $Dist, $Work | Out-Null

$Arguments = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--console",
    "--onefile",
    "--name", "NetPrism",
    "--paths", $Root,
    "--distpath", $Dist,
    "--workpath", $Work,
    "--specpath", $Work,
    "--add-data", "$Root\web;web",
    "--add-data", "$Root\examples\demo;examples\demo",
    "--add-data", "$Root\netprism\data;netprism\data",
    "--add-data", "$Root\schema;schema",
    # modules reached only through lazy imports inside the CLI/adapters
    "--hidden-import", "netprism.cli",
    "--hidden-import", "netprism.server",
    "--hidden-import", "netprism.core.astscan",
    "--hidden-import", "netprism.core.build",
    "--hidden-import", "netprism.core.discover",
    "--hidden-import", "netprism.core.groups",
    "--hidden-import", "netprism.core.ops",
    "--hidden-import", "netprism.core.sources",
    "--hidden-import", "netprism.core.synth",
    "--hidden-import", "netprism.metrics",
    "--hidden-import", "netprism.report",
    "--hidden-import", "netprism.algorithms",
    "--hidden-import", "netprism.algorithm_kb",
    "--hidden-import", "netprism.adapters.keras_adapter",
    # keep the lean build lean
    "--exclude-module", "torch",
    "--exclude-module", "torchvision",
    "--exclude-module", "onnx",
    "--exclude-module", "onnxruntime",
    "--exclude-module", "ml_dtypes",
    "--exclude-module", "keras",
    "--exclude-module", "tensorflow",
    "--exclude-module", "jax",
    "--exclude-module", "PySide6",
    "--exclude-module", "shiboken6",
    "--exclude-module", "netprism.gui",
    "--exclude-module", "pandas",
    "--exclude-module", "matplotlib",
    "--exclude-module", "PIL",
    "--exclude-module", "playwright",
    "--exclude-module", "pytest",
    "--exclude-module", "_pytest",
    "--exclude-module", "IPython",
    "--exclude-module", "tkinter"
)
if (Test-Path $Icon) {
    $Arguments += @("--icon", $Icon)
}
$Arguments += $Entry

Write-Host "Building lean Web EXE..." -ForegroundColor Cyan
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed" }

$Exe = Join-Path $Dist "NetPrism.exe"
if (-not (Test-Path $Exe)) { throw "Build finished but $Exe is missing" }
$SizeMb = [math]::Round((Get-Item $Exe).Length / 1MB, 1)
Write-Host "Built: $Exe ($SizeMb MB)" -ForegroundColor Green

if (-not $SkipSmoke) {
    # PyInstaller's onefile bootloader can fail to create its extraction dir when
    # the exe sits inside a sandboxed/redirected folder (seen in this workspace).
    # Smoke-test a copy from %TEMP% so the check measures the build, not the path.
    Write-Host "Smoke test (server + APIs):" -ForegroundColor Cyan
    $SmokeExe = Join-Path ([IO.Path]::GetTempPath()) "NetPrism-smoke-$([guid]::NewGuid().ToString('N').Substring(0, 8)).exe"
    Copy-Item $Exe $SmokeExe -Force
    try {
        & $SmokeExe --smoke
        if ($LASTEXITCODE -ne 0) { throw "Smoke test failed with exit code $LASTEXITCODE" }
    } finally {
        Remove-Item $SmokeExe -Force -ErrorAction SilentlyContinue
    }
    Write-Host "OK: frozen Web EXE smoke test passed." -ForegroundColor Green
}

$info = @"
NetPrism Windows build (Web mode)
mode: onefile console
exe: $Exe
size: $SizeMb MB
built: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
python: $Python
"@
$info | Set-Content -Path (Join-Path $Dist "BUILD_INFO.txt") -Encoding UTF8
exit 0
