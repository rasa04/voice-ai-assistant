"""
Local override for PyInstaller's contrib hook-webrtcvad.

`webrtcvad-wheels` provides the `webrtcvad` module, but not distribution
metadata under the `webrtcvad` name. The upstream contrib hook tries to call
copy_metadata("webrtcvad"), which crashes packaging with PackageNotFoundError.

This override keeps packaging stable and lets PyInstaller analyze the extension
module normally.
"""

hiddenimports: list[str] = []
datas: list[tuple[str, str]] = []
binaries: list[tuple[str, str]] = []
