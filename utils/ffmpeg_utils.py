import os
import shutil
from pathlib import Path


def get_ffmpeg_dir() -> str | None:
    """Return the directory containing ffmpeg.exe."""
    env_path = os.getenv("FFMPEG_PATH") or os.getenv("FFMPEG_BIN")
    if env_path:
        path = Path(env_path)
        if path.is_file():
            return str(path.parent)
        if (path / "ffmpeg.exe").exists() or (path / "ffmpeg").exists():
            return str(path)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return str(Path(ffmpeg).parent)

    winget_root = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if winget_root.exists():
        for pkg in winget_root.glob("Gyan.FFmpeg*"):
            for candidate in pkg.glob("**/bin/ffmpeg.exe"):
                return str(candidate.parent)

    for candidate in (
        Path("C:/ffmpeg/bin"),
        Path(os.environ.get("ProgramFiles", "")) / "ffmpeg" / "bin",
    ):
        if (candidate / "ffmpeg.exe").exists():
            return str(candidate)

    return None


def configure_pydub(ffmpeg_dir: str) -> None:
    bin_dir = Path(ffmpeg_dir)
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")

    from pydub import AudioSegment

    AudioSegment.converter = str(bin_dir / "ffmpeg.exe")
    AudioSegment.ffprobe = str(bin_dir / "ffprobe.exe")
