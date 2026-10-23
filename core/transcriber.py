import os

import httpx
from groq import Groq

from core.errors import UserFacingError, classify_upstream_error, safe_log
from utils.validation import GROQ_MAX_AUDIO_BYTES

SUPPORTED_LANGUAGES = {
    "urdu": "ur",
    "hindi": "hi",
    "english": "en",
}

GROQ_WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")


def get_client() -> Groq:
    """Create the Groq API client with explicit timeout and retry bounds."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise UserFacingError(
            "The transcription service is not configured. Please try again later."
        )

    return Groq(
        api_key=api_key,
        timeout=httpx.Timeout(120.0, connect=10.0),
        max_retries=2,
    )


def transcribe_chunk(
    chunk_path: str,
    language: str = "urdu",
    translate: bool = False,
) -> str:
    """Transcribe one audio chunk using Groq Whisper."""
    if language.lower() not in SUPPORTED_LANGUAGES:
        raise UserFacingError("That language is not supported.")

    if not os.path.isfile(chunk_path):
        raise UserFacingError("An audio segment was missing. Please try again.")

    size_bytes = os.path.getsize(chunk_path)
    if size_bytes <= 0:
        raise UserFacingError("An audio segment was empty. Please try another file.")
    if size_bytes >= GROQ_MAX_AUDIO_BYTES:
        raise UserFacingError(
            "A processed audio segment was too large for the transcription service. "
            "Please try a shorter video."
        )

    client = get_client()

    try:
        if translate:
            with open(chunk_path, "rb") as audio_file:
                result = client.audio.translations.create(
                    file=audio_file,
                    model=GROQ_WHISPER_MODEL,
                    response_format="json",
                    temperature=0.0,
                )
        else:
            with open(chunk_path, "rb") as audio_file:
                result = client.audio.transcriptions.create(
                    file=audio_file,
                    model=GROQ_WHISPER_MODEL,
                    language=SUPPORTED_LANGUAGES[language],
                    response_format="json",
                    temperature=0.0,
                )
    except Exception as exc:
        mapped = classify_upstream_error(exc)
        if mapped:
            raise mapped from exc
        safe_log("Transcription request failed", exc)
        raise UserFacingError(
            "Transcription failed because of an AI-service error. Please try again."
        ) from exc

    return result.text.strip()


def transcribe_all(
    chunks: list[str],
    language: str = "urdu",
    translate: bool = False,
) -> str:
    """Transcribe audio chunks. Temporary files are cleaned up by the caller."""
    if not chunks:
        raise UserFacingError("No audio was available to transcribe.")

    transcripts = []
    for i, chunk in enumerate(chunks, start=1):
        safe_log(f"Transcribing chunk {i}/{len(chunks)}")
        text = transcribe_chunk(
            chunk_path=chunk,
            language=language,
            translate=translate,
        )
        if text:
            transcripts.append(text)

    safe_log("Transcription complete")
    return " ".join(transcripts)
