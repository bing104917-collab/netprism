# Signing NetPrism

Public distribution needs a **trusted code-signing certificate**; a self-signed
certificate only works on machines that explicitly trust it.

## 0. Free open-source route (chosen)

SignPath Foundation signs open-source binaries in SignPath's HSM with a
certificate issued to SignPath Foundation; no local private key and no USB token.

Requirements:

- public GitHub repository with the MIT license and a released build;
- all build jobs run on GitHub-hosted runners;
- a **Code signing policy** on the project home page
  (`CODE_SIGNING_POLICY.md`, linked from `README.md`);
- approval by SignPath Foundation (not guaranteed; new projects may need
  verifiable reputation).

Steps:

1. Create a public GitHub repo named `netprism` and push this project.
2. Replace `YOUR_GITHUB_HANDLE` in `CODE_SIGNING_POLICY.md` with the maintainer
   account; keep the SignPath wording.
3. Apply at <https://signpath.org/apply> with the repository URL.
4. After approval, in SignPath.io create the project, link the GitHub trusted
   build system, create the signing policy and an artifact configuration for
   `NetPrismGUI.exe`.
5. Add repository variables/secrets used by
   `.github/workflows/windows-sign.yml`:
   - `SIGNPATH_API_TOKEN` (secret)
   - `SIGNPATH_ORGANIZATION_ID`
   - `SIGNPATH_PROJECT_SLUG`
   - `SIGNPATH_SIGNING_POLICY_SLUG`
   - `SIGNPATH_ARTIFACT_CONFIG_SLUG`
6. Create a tag (`v0.1.0`) or run the `windows-build-sign` workflow manually.
   It builds on Windows, uploads the unsigned artifact, submits it to SignPath,
   downloads the signed `NetPrismGUI.exe`, and uploads `NetPrismGUI-signed`.
7. Download the signed artifact and distribute that file. Do not sign a locally
   built exe and call it SignPath-signed: SignPath verifies the GitHub build
   origin.

Artifact configuration must constrain product name/version. `packaging/version_info.txt`
sets `ProductName=NetPrism`, `ProductVersion=0.1.0.0`; the workflow passes the
release tag as a signing parameter.

SignPath Foundation may reject a new project without public reputation. If that
happens, use Azure Trusted Signing or an OV/EV certificate (sections below);
self-signing is only for local testing.

## 1. Get a certificate

| Route | Suitable for | What you need |
|---|---|---|
| **Azure Trusted Signing** | individuals / small teams; no USB token | Azure subscription + identity validation + certificate profile |
| **OV/EV code-signing certificate** | commercial distribution | purchase from DigiCert/Sectigo/SSL.com/Certum; identity check; token/cloud HSM |
| **SignPath / OSS program** | public open-source project | public repository + application/approval |
| **Self-signed** | local testing only | nothing; does **not** remove SmartScreen for other people |

A code-signing certificate is not an SSL certificate. Since 2023 the private key
must live on a hardware token or cloud HSM for publicly trusted OV/EV certs.

## 2. Install `signtool`

Install the Windows SDK and select **Windows SDK Signing Tools for Desktop Apps**.
`signtool.exe` is usually at:

```
C:\Program Files (x86)\Windows Kits\10\bin\<version>\x64\signtool.exe
```

## 3. Sign the built GUI

Sign **after** PyInstaller, before copying to the desktop:

```powershell
# One-file GUI
.\scripts\sign_exe.ps1 -Exe .\dist\gui\NetPrismGUI.exe -PfxPath .\codesign.pfx

# Onedir GUI
.\scripts\sign_exe.ps1 -Exe .\dist\gui\NetPrismGUI\NetPrismGUI.exe -Thumbprint <cert-sha1>

# Certificate already installed / USB token
.\scripts\sign_exe.ps1 -Exe .\dist\gui\NetPrismGUI.exe -Subject "Your Publisher Name"

# Azure Trusted Signing
.\scripts\sign_exe.ps1 -Exe .\dist\gui\NetPrismGUI.exe `
  -AzureDlib .\Azure.CodeSigning.Dlib.dll -AzureMetadata .\metadata.json
```

The script uses SHA-256, a timestamp server, verifies the result with
`signtool verify /pa /v`, and prints `Get-AuthenticodeSignature`.

Do not commit a `.pfx` or paste its password into chat. Keep it local and pass
the path; the script prompts for the password if needed.

## 4. Verify

```powershell
Get-AuthenticodeSignature .\dist\gui\NetPrismGUI.exe |
  Format-List Status,StatusMessage,SignerCertificate,TimeStamperCertificate
```

`Status` should be `Valid`. A new OV certificate may still accumulate SmartScreen
reputation over time; EV/Trusted Signing generally builds trust faster.

## 5. Timestamp

Never omit `/tr` + `/td SHA256`: a timestamp keeps the signature valid after the
certificate expires. Default servers:

- `http://timestamp.digicert.com`
- `http://timestamp.sectigo.com`
- Azure Trusted Signing: `http://timestamp.acs.microsoft.com`

## 6. Self-signed (tests only)

```powershell
$cert = New-SelfSignedCertificate -Type CodeSigningCert `
  -Subject "CN=NetPrism Local Signing" `
  -CertStoreLocation Cert:\CurrentUser\My -KeyUsage DigitalSignature

Set-AuthenticodeSignature -FilePath .\dist\gui\NetPrismGUI.exe `
  -Certificate $cert -HashAlgorithm SHA256 `
  -TimestampServer "http://timestamp.digicert.com"
```

Other people will still see “Unknown publisher” unless they install the
certificate into Trusted Publisher/Trusted Root.
