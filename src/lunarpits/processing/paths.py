"""Pure path conversion helpers used by Windows-to-WSL workflows."""

from __future__ import annotations

from pathlib import PureWindowsPath


def windows_path_to_wsl(path: str | PureWindowsPath) -> str:
    """Convert an absolute Windows path to its WSL ``/mnt/<drive>`` form.

    Parameters
    ----------
    path:
        Absolute Windows path such as ``C:\\Users\\name\\file.tif``.

    Raises
    ------
    ValueError
        If the path does not include a Windows drive.
    """
    win_path = PureWindowsPath(path)
    drive = win_path.drive.rstrip(":").lower()
    if not drive:
        raise ValueError(f"Windows path must include a drive: {path!s}")

    parts = [part for part in win_path.parts[1:] if part not in ("\\", "/")]
    suffix = "/".join(parts)
    return f"/mnt/{drive}/{suffix}" if suffix else f"/mnt/{drive}"

