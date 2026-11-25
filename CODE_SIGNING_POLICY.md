# Code signing policy

**Free code signing provided by [SignPath.io](https://about.signpath.io), certificate by [SignPath Foundation](https://signpath.org).**

Status: application/approval pending. Signed Windows releases are produced only
by the GitHub Actions workflow from this repository; no local private key is
used.

## Team roles

- Committers and reviewers: `@bing104917-collab`
- Approvers: `@bing104917-collab`

Replace `YOUR_GITHUB_HANDLE` with the actual GitHub account/team before
submitting the SignPath application.

## Build and release

- Source: this public GitHub repository (MIT).
- Build: `.github/workflows/windows-sign.yml`, GitHub-hosted Windows runner.
- Artifact: `NetPrismGUI.exe`, built with `scripts/build_gui.ps1 -OneFile`.
- Signing: SignPath.io artifact configuration for `NetPrismGUI.exe` only.
- Every release requires an explicit approver action in SignPath.

## Privacy

This program does not transfer model files, paths, telemetry or usage data to
networked systems unless specifically requested by the user. Model parsing and
the native GUI run locally. The application does not open a local HTTP server
or a browser.

## Uninstallation

The GUI executable is a single file; delete the executable (and its optional
`netprism_gui_runtime` extraction directory) to uninstall it. No installer or
system service is installed.

## Scope

Only binaries built from this repository are signed. Upstream open-source
libraries bundled by PyInstaller (PyTorch, PySide6, ONNX, etc.) remain under
their own licenses and are not re-signed as NetPrism code.
