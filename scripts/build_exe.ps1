param(
    [ValidateSet("onedir", "onefile")]
    [string]$Mode = "onedir",
    [string]$Name = "",
    [string]$RuntimeTmpDir = "",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $project
New-Item -ItemType Directory -Force -Path (Join-Path $project "build") | Out-Null

$python = $env:NETPRISM_PYTHON
if (-not $python) {
    $python = "C:\Users\hrbkcy2523\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
}
if (-not (Test-Path $python)) {
    $python = "python"
}

if (-not $Name) {
    $Name = if ($Mode -eq "onefile") { "NetPrism-Portable" } else { "NetPrism" }
}

if ($Mode -eq "onefile" -and -not $RuntimeTmpDir) {
    # Some locked-down Windows profiles deny the bootloader's default %TEMP%
    # extraction. Extracting beside the exe (relative to cwd) is reliable when
    # the app is placed in a writable folder.
    $RuntimeTmpDir = "netprism_runtime"
}

$icon = Join-Path $project "packaging\netprism.ico"
$entry = Join-Path $project "packaging\launcher.py"
if (-not (Test-Path $icon)) {
    throw "Missing icon: $icon. Run packaging/make_icon.py first."
}

# onnx's native extension links a hashed MSVC runtime DLL that lives in the
# sibling onnx.libs directory (not inside the onnx package). PyInstaller does
# not collect it automatically, so add it explicitly.
$onnxLibs = (& $python -c "import onnx, pathlib; print(pathlib.Path(onnx.__file__).resolve().parent.parent / 'onnx.libs')" 2>$null).Trim()
$extraData = @()
if ($onnxLibs -and (Test-Path $onnxLibs)) {
    $extraData += @("--add-data", "$onnxLibs;onnx.libs")
}
$mlLibs = (& $python -c "import ml_dtypes, pathlib; print(pathlib.Path(ml_dtypes.__file__).resolve().parent.parent / 'ml_dtypes.libs')" 2>$null).Trim()
if ($mlLibs -and (Test-Path $mlLibs)) {
    $extraData += @("--add-data", "$mlLibs;ml_dtypes.libs")
}

if (-not $SkipTests) {
    Write-Host "== running tests before packaging ==" -ForegroundColor Cyan
    & $python -m pytest tests -q
    if ($LASTEXITCODE -ne 0) { throw "tests failed; not packaging" }
}

Write-Host "== building $Mode $Name ==" -ForegroundColor Cyan
$args = @(
    "-m", "PyInstaller",
    "--noconfirm", "--clean", "--log-level", "INFO",
    "--name", $Name,
    "--icon", $icon,
    "--paths", $project,
    "--distpath", (Join-Path $project "dist"),
    "--workpath", (Join-Path $project "build"),
    "--specpath", (Join-Path $project "build"),
    "--add-data", "$project\web;web",
    "--add-data", "$project\examples\demo;examples\demo",
    "--add-data", "$project\examples\multiformat;examples\multiformat",
    "--add-data", "$project\samples;samples",
    "--add-data", "$project\schema;schema",
    "--add-data", "$project\netprism\data;netprism\data",
    "--add-data", "$project\README.md;.",
    "--collect-submodules", "torch.fx",
    "--collect-submodules", "onnx",
    "--hidden-import", "onnx",
    "--hidden-import", "onnx.shape_inference",
    "--hidden-import", "onnx.helper",
    "--hidden-import", "netprism.adapters",
    "--hidden-import", "netprism.adapters.onnx_adapter",
    "--hidden-import", "netprism.adapters.torchscript_adapter",
    "--hidden-import", "netprism.adapters.keras_adapter",
    "--hidden-import", "netprism.metrics",
    "--hidden-import", "netprism.report",
    "--hidden-import", "netprism.compare",
    "--hidden-import", "netprism.algorithms",
    "--hidden-import", "netprism.algorithm_kb",
    "--hidden-import", "torch.fx",
    "--hidden-import", "torch.fx.graph_module",
    "--hidden-import", "torch.fx.interpreter",
    "--hidden-import", "torch.fx._symbolic_trace",
    "--hidden-import", "torch.fx.passes.shape_prop",
    "--exclude-module", "playwright",
    "--exclude-module", "pytest",
    "--exclude-module", "_pytest",
    "--exclude-module", "jsonschema",
    "--exclude-module", "jsonschema_specifications",
    "--exclude-module", "referencing",
    "--exclude-module", "pandas",
    "--exclude-module", "matplotlib",
    "--exclude-module", "IPython",
    "--exclude-module", "notebook",
    "--exclude-module", "jupyter",
    "--exclude-module", "tkinter",
    "--exclude-module", "lxml",
    "--exclude-module", "openpyxl",
    "--exclude-module", "docx",
    "--exclude-module", "pptx"
)
$args += $extraData
if ($Mode -eq "onefile") {
    $args += "--onefile"
    if ($RuntimeTmpDir) {
        # Relative paths are resolved against the current working directory
        # (usually the exe's folder when launched from Explorer). This makes
        # the portable exe independent of a locked-down %TEMP%.
        $args += "--runtime-tmpdir"
        $args += $RuntimeTmpDir
    }
} else {
    $args += "--onedir"
}
$args += $entry

& $python @args
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

if ($Mode -eq "onefile") {
    $exe = Join-Path $project "dist\$Name.exe"
    $size = [math]::Round((Get-Item $exe).Length / 1MB, 1)
    Write-Host "BUILT $exe ($size MB)" -ForegroundColor Green
} else {
    $exe = Join-Path $project "dist\$Name\$Name.exe"
    $folder = Join-Path $project "dist\$Name"
    $size = [math]::Round((Get-ChildItem $folder -Recurse -File | Measure-Object Length -Sum).Sum / 1MB, 1)
    Write-Host "BUILT $exe (folder $size MB)" -ForegroundColor Green
}

$info = @"
NetPrism Windows build
mode: $Mode
name: $Name
built: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
python: $python
exe: $exe
"@
$info | Set-Content -Path (Join-Path $project "dist\BUILD_INFO.txt") -Encoding UTF8
