"""User-safe errors and redacted logging. Do not put secrets in messages."""


class UserFacingError(Exception):
    """Exception whose string is safe to show in the Streamlit UI."""


class RateLimitedError(UserFacingError):
    """Provider returned HTTP 429 or an equivalent rate-limit error."""


class UpstreamTimeoutError(UserFacingError):
    """An external API call exceeded its timeout."""


_RATE_LIMIT_TYPES = {"RateLimitError", "TooManyRequestsError"}
_TIMEOUT_TYPES = {"APITimeoutError", "TimeoutError", "ReadTimeout", "ConnectTimeout"}


def _exception_names(exc: BaseException) -> set[str]:
    names = {type(exc).__name__}
    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, BaseException):
        names.add(type(cause).__name__)
    return names


def classify_upstream_error(exc: BaseException) -> UserFacingError | None:
    """Map provider SDK exceptions to a safe user-facing error, if recognized."""
    names = _exception_names(exc)
    if names & _RATE_LIMIT_TYPES:
        return RateLimitedError(
            "The AI service is busy right now. Please wait a moment and try again."
        )
    if names & _TIMEOUT_TYPES:
        return UpstreamTimeoutError(
            "The request timed out. Please try again with a shorter video."
        )

    status = getattr(exc, "status_code", None)
    if status == 429:
        return RateLimitedError(
            "The AI service is busy right now. Please wait a moment and try again."
        )
    return None


def safe_log(message: str, exc: BaseException | None = None) -> None:
    """Log a diagnostic line without exception bodies, paths, or secrets."""
    if exc is None:
        print(f"Vidora: {message}")
        return
    print(f"Vidora: {message} ({type(exc).__name__})")
