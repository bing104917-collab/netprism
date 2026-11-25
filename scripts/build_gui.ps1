# Build the native PySide6 NetPrism GUI executable.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\build_gui.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\build_gui.ps1 -Python "C:\Python312\python.exe"
#
# The resulting windowed executable is dist\gui\NetPrismGUI\NetPrismGUI.exe.
# It includes examples\demo but never starts an HTTP server or web engine.

param(
    [string]$Python = "python",
    [switch]$OneFile
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Launcher = Join-Path $Root "packaging\gui_launcher.py"
$Icon = Join-Path $Root "packaging\netprism.ico"
$Dist = Join-Path $Root "dist\gui"
$Work = Join-Path $Root "build\gui"
$Spec = Join-Path $Work "NetPrismGUI.spec"

if (-not (Test-Path $Launcher)) { throw "GUI launcher not found: $Launcher" }
if (-not (Test-Path (Join-Path $Root "examples\demo"))) { throw "examples\demo not found" }

Write-Host "Checking PyInstaller..."
& $Python -c "import PyInstaller, sys; print('PyInstaller', PyInstaller.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller is not installed. Run: $Python -m pip install pyinstaller"
}

New-Item -ItemType Directory -Force -Path $Dist | Out-Null
New-Item -ItemType Directory -Force -Path $Work | Out-Null

# Optional native-extension side libraries for tracing / format adapters.
$ExtraData = @()
$OnnxLibs = (& $Python -c "import onnx, pathlib; print(pathlib.Path(onnx.__file__).resolve().parent.parent / 'onnx.libs')" 2>$null)
if ($OnnxLibs) { $OnnxLibs = $OnnxLibs.Trim() }
if ($OnnxLibs -and (Test-Path $OnnxLibs)) {
    $ExtraData += @("--add-data", "$OnnxLibs;onnx.libs")
}
$MlLibs = (& $Python -c "import ml_dtypes, pathlib; print(pathlib.Path(ml_dtypes.__file__).resolve().parent.parent / 'ml_dtypes.libs')" 2>$null)
if ($MlLibs) { $MlLibs = $MlLibs.Trim() }
if ($MlLibs -and (Test-Path $MlLibs)) {
    $ExtraData += @("--add-data", "$MlLibs;ml_dtypes.libs")
}

$Arguments = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--windowed",
    "--name", "NetPrismGUI",
    "--paths", $Root,
    "--distpath", $Dist,
    "--workpath", $Work,
    "--specpath", (Split-Path -Parent $Spec),
    "--add-data", "$Root\examples\demo;examples\demo",
    "--add-data", "$Root\netprism\data;netprism\data",
    "--add-data", "$Root\schema;schema",
    "--collect-submodules", "netprism",
    "--collect-submodules", "torch.fx",
    "--collect-submodules", "onnx",
    "--hidden-import", "netprism.cli",
    "--hidden-import", "netprism.adapters",
    "--hidden-import", "netprism.adapters.onnx_adapter",
    "--hidden-import", "netprism.adapters.torchscript_adapter",
    "--hidden-import", "netprism.adapters.keras_adapter",
    "--hidden-import", "netprism.metrics",
    "--hidden-import", "netprism.report",
    "--hidden-import", "netprism.algorithms",
    "--hidden-import", "netprism.algorithm_kb",
    "--hidden-import", "onnx",
    "--hidden-import", "onnx.shape_inference",
    "--hidden-import", "onnx.helper",
    "--hidden-import", "torch.fx",
    "--hidden-import", "torch.fx.graph_module",
    "--hidden-import", "torch.fx.interpreter",
    "--hidden-import", "torch.fx._symbolic_trace",
    "--hidden-import", "torch.fx.passes.shape_prop",
    "--exclude-module", "netprism.server",
    "--exclude-module", "playwright",
    "--exclude-module", "pytest",
    "--exclude-module", "_pytest",
    "--exclude-module", "matplotlib",
    "--exclude-module", "pandas",
    "--exclude-module", "IPython",
    "--exclude-module", "jupyter",
    "--exclude-module", "tkinter"
)
$Arguments += $ExtraData
if (Test-Path $Icon) {
    $Arguments += @("--icon", $Icon)
}
$VersionFile = Join-Path $Root "packaging\version_info.txt"
if (Test-Path $VersionFile) {
    $Arguments += @("--version-file", $VersionFile)
}
if ($OneFile) {
    $Arguments += @("--onefile", "--runtime-tmpdir", "netprism_gui_runtime")
}
$Arguments += $Launcher

Write-Host "Building native GUI..."
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed" }

$Exe = if ($OneFile) { Join-Path $Dist "NetPrismGUI.exe" } else { Join-Path $Dist "NetPrismGUI\NetPrismGUI.exe" }
if (-not (Test-Path $Exe)) { throw "Build finished but executable not found: $Exe" }

Write-Host ""
Write-Host "Built: $Exe"
Write-Host "Smoke test (offscreen):"
$SmokeScript = Join-Path $Work "smoke_gui.py"
@'
import os
import subprocess
import sys

exe = os.environ["NETPRISM_SMOKE_EXE"]
env = dict(os.environ)
env["QT_QPA_PLATFORM"] = "offscreen"
proc = subprocess.run([exe, "--smoke"], env=env, capture_output=True, timeout=180)
sys.stdout.write(proc.stdout.decode("utf-8", "replace"))
sys.stderr.write(proc.stderr.decode("utf-8", "replace"))
raise SystemExit(proc.returncode)
'@ | Set-Content -Path $SmokeScript -Encoding ASCII
$env:NETPRISM_SMOKE_EXE = $Exe
& $Python $SmokeScript
$code = $LASTEXITCODE
Remove-Item Env:\NETPRISM_SMOKE_EXE -ErrorAction SilentlyContinue
if ($code -ne 0) { throw "GUI smoke test failed with exit code $code" }
Write-Host "OK: native GUI smoke test passed."
exit 0
