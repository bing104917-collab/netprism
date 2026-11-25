# Sign a NetPrism executable with signtool.
#
# Examples:
#   .\scripts\sign_exe.ps1 -Exe .\dist\gui\NetPrismGUI.exe -PfxPath .\codesign.pfx
#   .\scripts\sign_exe.ps1 -Exe .\dist\gui\NetPrismGUI.exe -Subject "Your Publisher Name"
#   .\scripts\sign_exe.ps1 -Exe .\dist\gui\NetPrismGUI.exe `
#       -AzureDlib .\Azure.CodeSigning.Dlib.dll -AzureMetadata .\metadata.json
#
# The password is prompted if the PFX is encrypted; do not put it in the command
# line or in source control.

param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [string]$PfxPath,
    [System.Security.SecureString]$PfxPassword,
    [string]$Thumbprint,
    [string]$Subject,
    [string]$AzureDlib,
    [string]$AzureMetadata,
    [string]$Timestamp = "http://timestamp.digicert.com",
    [switch]$NoTimestamp
)

$ErrorActionPreference = "Stop"

function Find-SignTool {
    $command = Get-Command signtool.exe -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $roots = @(
        "C:\Program Files (x86)\Windows Kits\10\bin",
        "C:\Program Files\Windows Kits\10\bin"
    )
    foreach ($root in $roots) {
        if (-not (Test-Path $root)) { continue }
        $candidate = Get-ChildItem $root -Recurse -Filter signtool.exe -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match '\\x64\\' } |
            Sort-Object FullName -Descending |
            Select-Object -First 1
        if ($candidate) { return $candidate.FullName }
    }
    throw "signtool.exe not found. Install the Windows SDK 'Signing Tools for Desktop Apps'."
}

if (-not (Test-Path -LiteralPath $Exe)) {
    throw "Executable not found: $Exe"
}
$Exe = (Resolve-Path -LiteralPath $Exe).Path
$signTool = Find-SignTool

$SignArgs = @("sign", "/fd", "SHA256", "/v")
$usingAzure = $AzureDlib -and $AzureMetadata
if ($usingAzure) {
    if (-not (Test-Path $AzureDlib)) { throw "Azure dlib not found: $AzureDlib" }
    if (-not (Test-Path $AzureMetadata)) { throw "Azure metadata not found: $AzureMetadata" }
    $SignArgs += @("/dlib", (Resolve-Path $AzureDlib).Path, "/dmdf", (Resolve-Path $AzureMetadata).Path)
    $Timestamp = "http://timestamp.acs.microsoft.com"
}
elseif ($PfxPath) {
    if (-not (Test-Path -LiteralPath $PfxPath)) { throw "PFX not found: $PfxPath" }
    if (-not $PfxPassword) {
        $PfxPassword = Read-Host "PFX password" -AsSecureString
    }
    $plain = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [Runtime.InteropServices.Marshal]::SecureStringToBSTR($PfxPassword)
    )
    $SignArgs += @("/f", (Resolve-Path -LiteralPath $PfxPath).Path, "/p", $plain)
}
elseif ($Thumbprint) {
    $SignArgs += @("/sha1", $Thumbprint)
}
elseif ($Subject) {
    $SignArgs += @("/n", $Subject)
}
else {
    throw "Provide -PfxPath, -Thumbprint, -Subject, or -AzureDlib/-AzureMetadata."
}

if (-not $NoTimestamp) {
    $SignArgs += @("/tr", $Timestamp, "/td", "SHA256")
}
$SignArgs += $Exe

Write-Host "Signing: $Exe"
Write-Host "  signer: $(if ($Subject) { $Subject } elseif ($Thumbprint) { $Thumbprint } elseif ($PfxPath) { $PfxPath } else { 'Azure Trusted Signing' })"
& $signTool @SignArgs
if ($LASTEXITCODE -ne 0) {
    throw "signtool failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Verifying..."
& $signTool verify /pa /v $Exe
if ($LASTEXITCODE -ne 0) {
    throw "signature verification failed"
}

Get-AuthenticodeSignature -LiteralPath $Exe |
    Format-List Status, StatusMessage, SignerCertificate, TimeStamperCertificate
