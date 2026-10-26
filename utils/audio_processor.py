import logging
import os
from pathlib import Path

import yt_dlp
from pydub import AudioSegment
import streamlit as st
import json
import subprocess
from pathlib import Path

from pydub import AudioSegment

from core.errors import UserFacingError, safe_log
import tempfile
from core.errors import UserFacingError, safe_log
from utils.validation import GROQ_MAX_AUDIO_BYTES, ydl_common_opts

MAX_CHUNK_MB = 18


def download_youtube_audio(url: str, work_dir: str) -> str:
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    output_path = os.path.join(work_dir, "source_audio.%(ext)s")
    cookies_file = None

    ydl_opts = {
        **ydl_common_opts(),
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
    }

    try:
        # Read cookie contents from Streamlit Secrets.
        cookies = st.secrets.get("YOUTUBE_COOKIES", "")

        if cookies.strip():
            fd, cookies_file = tempfile.mkstemp(
                prefix="vidora_cookies_",
                suffix=".txt",
            )

            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(cookies.strip() + "\n")

            ydl_opts["cookiefile"] = cookies_file

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            if not info:
                raise RuntimeError("No video information returned.")

            wav_path = os.path.splitext(
                ydl.prepare_filename(info)
            )[0] + ".wav"

        if not os.path.isfile(wav_path):
            raise RuntimeError("WAV output was not created.")

        return wav_path
    except Exception as exc:

        logging.getLogger(__name__).exception(
        "YouTube audio extraction failed"
        )

        safe_log(
        f"YouTube audio extraction failed: "
        f"{type(exc).__name__}: {exc}"
        )

        raise UserFacingError(
        "Vidora could not download this YouTube video. "
        "Check the server logs for details."
        )  from exc

    finally:
        if cookies_file and os.path.exists(cookies_file):
            try:
                os.remove(cookies_file)
            except OSError:
                safe_log("Could not remove temporary cookie file")


def convert_to_wav(input_path: str, work_dir: str) -> str:
    """Extract audio from uploaded media and convert it to mono, 16 kHz WAV."""
    input_file = Path(input_path)

    if not input_file.is_file() or input_file.stat().st_size == 0:
        raise UserFacingError(
            "The uploaded file is empty or could not be found."
        )

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_streams",
                "-of", "json",
                str(input_file),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )

        streams = json.loads(result.stdout).get("streams", [])
        audio_streams = [
            stream for stream in streams
            if stream.get("codec_type") == "audio"
        ]

        if not audio_streams:
            raise UserFacingError(
                "This video does not contain a readable audio track. "
                "Please upload a video with audio or an audio file."
            )

        output_path = str(Path(work_dir) / "converted.wav")

        audio = AudioSegment.from_file(str(input_file))
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(output_path, format="wav")

        if not Path(output_path).is_file() or Path(output_path).stat().st_size == 0:
            raise UserFacingError("Audio conversion produced an empty file.")

        safe_log("Uploaded media converted successfully")
        return output_path

    except UserFacingError:
        raise
    except subprocess.TimeoutExpired as exc:
        raise UserFacingError(
            "Checking the uploaded video took too long. Try a shorter file."
        ) from exc
    except subprocess.CalledProcessError as exc:
        safe_log(f"FFprobe failed with exit code {exc.returncode}")
        safe_log(f"FFprobe error: {exc.stderr[-1000:]}")
        raise UserFacingError(
            "Vidora could not inspect this video. Try exporting it as MP4 "
            "with AAC audio and uploading it again."
        ) from exc
    except Exception as exc:
        import logging
        logging.getLogger(__name__).exception(
            "Uploaded media audio conversion failed"
        )
        raise UserFacingError(
            "Vidora could not extract audio from this file. "
            "Try an MP4 video with AAC audio."
        ) from exc


def chunk_data(wav_path: str, work_dir: str, max_chunk_mb: int = MAX_CHUNK_MB) -> list[str]:
    """Split audio into WAV chunks smaller than the Groq upload limit."""
    audio = AudioSegment.from_wav(wav_path)

    if len(audio) <= 0 or not audio.raw_data:
        raise UserFacingError("The audio file is empty or could not be read.")

    target_bytes = max_chunk_mb * 1024 * 1024
    bytes_per_ms = len(audio.raw_data) / len(audio)
    if bytes_per_ms <= 0:
        raise UserFacingError("The audio file is empty or could not be read.")

    chunk_ms = max(1, int((target_bytes * 0.9) / bytes_per_ms))
    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]
        chunk_path = os.path.join(work_dir, f"chunk_{i}.wav")
        chunk.export(chunk_path, format="wav")

        size_bytes = os.path.getsize(chunk_path)
        if size_bytes <= 0 or size_bytes >= GROQ_MAX_AUDIO_BYTES:
            try:
                os.remove(chunk_path)
            except OSError:
                safe_log("Could not remove an oversized audio chunk")
            raise UserFacingError(
                "A processed audio segment was too large for the transcription service. "
                "Please try a shorter video."
            )

        chunks.append(chunk_path)
        safe_log(f"Created audio chunk {i + 1}")

    return chunks


def process_input(source: str, work_dir: str) -> list[str]:
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    if source.startswith(("http://", "https://")):
        safe_log("Downloading YouTube audio")
        wav_path = download_youtube_audio(source, work_dir)
    else:
        safe_log("Converting local audio")
        wav_path = convert_to_wav(source, work_dir)

    safe_log("Splitting audio into chunks")
    chunks = chunk_data(wav_path, work_dir)
    safe_log(f"Audio ready with {len(chunks)} chunk(s)")
    return chunks

