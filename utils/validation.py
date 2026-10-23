"""Input validation used before expensive download or transcription work."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from core.errors import UserFacingError

ALLOWED_EXTENSIONS = {
    "mp3",
    "mp4",
    "m4a",
    "wav",
    "webm",
    "mpeg",
    "mpga",
    "ogg",
    "flac",
}

ALLOWED_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "www.youtu.be",
    "music.youtube.com",
    "www.music.youtube.com",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
}

# Groq Whisper historically rejects files at or above 25 MB.
GROQ_MAX_AUDIO_BYTES = 25 * 1024 * 1024
BLOCKED_LOCAL_NAMES = {".env", ".env.local", "secrets.toml", "credentials.json"}


def is_http_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def assert_youtube_url(source: str) -> None:
    parsed = urlparse(source)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or host not in ALLOWED_YOUTUBE_HOSTS:
        raise UserFacingError("Please paste a valid YouTube URL.")


def normalize_duration(value) -> float:
    if value is None:
        raise UserFacingError(
            "The video's duration could not be verified, so it cannot be processed."
        )
    try:
        duration = float(value)
    except (TypeError, ValueError) as exc:
        raise UserFacingError(
            "The video's duration could not be verified, so it cannot be processed."
        ) from exc
    if duration <= 0 or duration != duration:  # NaN
        raise UserFacingError(
            "The video's duration could not be verified, so it cannot be processed."
        )
    return duration


def assert_duration_within_limit(duration: float, max_minutes: int) -> None:
    if duration > max_minutes * 60:
        raise UserFacingError(
            f"Please choose a video no longer than {max_minutes} minutes."
        )


def assert_single_youtube_video(info: dict) -> None:
    info_type = str(info.get("_type") or "video").lower()
    if info_type == "playlist":
        raise UserFacingError(
            "Playlists are not supported. Please paste a link to a single YouTube video."
        )


def extension_of(path_or_name: str) -> str:
    return Path(path_or_name).suffix.lower().lstrip(".")


def assert_allowed_extension(path_or_name: str) -> None:
    if extension_of(path_or_name) not in ALLOWED_EXTENSIONS:
        raise UserFacingError("This file type is not supported.")


def assert_file_size(size_bytes: int | None, max_mb: int) -> None:
    if size_bytes is None or size_bytes <= 0:
        raise UserFacingError("The selected file is empty.")
    if size_bytes > max_mb * 1024 * 1024:
        raise UserFacingError(f"Please upload a file smaller than {max_mb} MB.")


def looks_like_media(header: bytes) -> bool:
    """Best-effort sniff of common audio/video containers. Not a full parser."""
    if not header or len(header) < 4:
        return False
    if header.startswith(b"RIFF") and b"WAVE" in header[:16]:
        return True
    if header.startswith(b"ID3") or header.startswith(b"fLaC") or header.startswith(b"OggS"):
        return True
    if header.startswith(b"\x1a\x45\xdf\xa3"):
        return True
    if len(header) >= 8 and header[4:8] == b"ftyp":
        return True
    if header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
        return True
    return False


def assert_media_header(header: bytes) -> None:
    if not looks_like_media(header):
        raise UserFacingError("This file type is not supported.")


def assert_safe_local_media(path: Path, max_mb: int) -> None:
    if not path.is_file():
        raise UserFacingError("The specified local file was not found.")
    if path.name.lower() in BLOCKED_LOCAL_NAMES:
        raise UserFacingError("This file type is not supported.")
    assert_allowed_extension(path.name)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise UserFacingError("The specified local file was not found.") from exc
    assert_file_size(size, max_mb)
    try:
        with path.open("rb") as handle:
            header = handle.read(16)
    except OSError as exc:
        raise UserFacingError("The specified local file could not be read.") from exc
    assert_media_header(header)


def ydl_common_opts() -> dict:
    """Shared yt-dlp options: no playlist, bounded socket timeout, quiet logs."""
    return {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 2,
        "fragment_retries": 2,
    }
