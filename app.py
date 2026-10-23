import base64
import html
import logging
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
import yt_dlp
from dotenv import load_dotenv

load_dotenv()


def _apply_secret(name: str) -> None:
    try:
        value = st.secrets.get(name)
    except (AttributeError, FileNotFoundError, Exception):
        return
    if value:
        os.environ.setdefault(name, str(value))


for _secret_name in (
    "GROQ_API_KEY",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "QDRANT_COLLECTION",
    "VIDORA_DEBUG",
):
    _apply_secret(_secret_name)


_DEBUG = os.getenv("VIDORA_DEBUG", "").lower() in ("1", "true", "yes")

log = logging.getLogger("framewise")
if not log.handlers:  # Streamlit re-runs this file on every interaction
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(_handler)
    log.setLevel(logging.DEBUG if _DEBUG else logging.INFO)
    log.propagate = False


@st.cache_resource
def _log_environment() -> bool:
    """Log the runtime once per server process, not on every rerun."""
    log.info("python=%s yt-dlp=%s", sys.version.split()[0], yt_dlp.version.__version__)
    log.info(
        "deno on PATH=%s | deno in venv=%s | node=%s | ffmpeg=%s",
        shutil.which("deno"),
        os.path.exists(os.path.join(sys.prefix, "bin", "deno")),
        shutil.which("node"),
        shutil.which("ffmpeg"),
    )
    return True


_log_environment()


def _log_exc(context: str, exc: Exception) -> None:
    """Log an error with its root cause. Detail only when VIDORA_DEBUG is on."""
    cause = exc.__cause__ or exc.__context__
    cause_txt = f" (cause: {type(cause).__name__})" if cause else ""
    detail = (
        str(exc).replace("\n", " ")[:300] if _DEBUG else "set VIDORA_DEBUG=1 for detail"
    )
    log.error(
        "%s: %s%s: %s", context, type(exc).__name__, cause_txt, detail, exc_info=_DEBUG
    )


from core.errors import UserFacingError, safe_log
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question
from core.summarizer import summarize, generate_title
from core.transcriber import transcribe_all
from core.vector_store import session_collection_name
from utils.audio_processor import process_input
from utils.validation import (
    assert_duration_within_limit,
    assert_file_size,
    assert_media_header,
    assert_safe_local_media,
    assert_single_youtube_video,
    assert_youtube_url,
    is_http_url,
    normalize_duration,
    ydl_common_opts,
)

MAX_JOBS_PER_SESSION = 2
MAX_QUESTIONS_PER_SESSION = 10
MAX_UPLOAD_MB = 25
MAX_VIDEO_MINUTES = 15
ALLOWED_EXTENSIONS = ["mp3", "mp4", "m4a", "wav", "webm", "mpeg", "mpga", "ogg", "flac"]

# ── Page config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Framewise",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Visual theme
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400..800&family=Newsreader:ital,opsz,wght@0,6..72,400..600;1,6..72,400..600&display=swap');
:root {
  --paper: #EDF0F4; --sheet: #FFFFFF; --sheet-2: #F5F7FA;
  --ink: #182033; --ink-2: #444F64; --ink-3: #667389;
  --line: #E1E6ED; --line-2: #C9D1DC;
  --blue: #3B4CE0; --blue-ink: #2433B8; --blue-soft: #E4E8FD;
  --rec: #E5484D; --ok: #1B7F5F; --amber: #B7791F; --bad: #B42318;
  --font-ui: 'Bricolage Grotesque', 'Segoe UI', system-ui, -apple-system, sans-serif;
  --font-read: 'Newsreader', 'Iowan Old Style', Georgia, serif;
  color-scheme: light;
}
html, body, .stApp, [data-testid="stAppViewContainer"] { background: var(--paper) !important; color: var(--ink); }
.stApp { font-family: var(--font-ui) !important; }
.stApp button, .stApp input, .stApp textarea { font-family: var(--font-ui); }
[data-testid="stHeader"] { background: transparent; }
footer, #MainMenu, [data-testid="stMainMenu"], [data-testid="stAppDeployButton"], .stDeployButton { display: none; }
.block-container { max-width: 1120px; padding-top: 2.2rem; padding-bottom: 5rem; }
.stApp :focus-visible { outline: 2px solid var(--blue); outline-offset: 2px; }
/* ── Sidebar ── */
[data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child { background: var(--sheet) !important; }
[data-testid="stSidebar"] { border-right: 1px solid var(--line-2); }
[data-testid="stSidebar"] hr { margin: 1rem 0; border: none; border-top: 1px solid var(--line); }
.v-wordmark { display: flex; align-items: center; gap: 0.7rem; font-weight: 700; font-size: 1.15rem; letter-spacing: -0.01em; color: var(--ink); }
.v-wordmark-icon { position: relative; flex: none; width: 2.3rem; height: 2.3rem; }
.v-wordmark-icon img { display: block; width: 100%; height: 100%; margin: 0; }
.v-wordmark-icon::after { content: ""; position: absolute; right: -3px; top: -3px; width: 8px; height: 8px; border-radius: 50%; background: var(--rec); border: 2px solid var(--sheet); box-sizing: content-box; }
.v-tagline { font-size: 0.78rem; font-weight: 500; color: var(--ink-3); }
.v-sidebar-label { margin: 1.2rem 0 0.4rem; font-size: 0.84rem; font-weight: 700; color: var(--ink-2); }
.v-footer { margin-top: 1.6rem; font-size: 0.78rem; color: var(--ink-3); }
.v-usage { margin: 1.2rem 0 0.4rem; }
.v-usage-title { margin-bottom: 0.4rem; font-size: 0.84rem; font-weight: 700; color: var(--ink-2); }
.v-usage-row { display: flex; justify-content: space-between; align-items: baseline; margin: 0.65rem 0 0.3rem; font-size: 0.88rem; color: var(--ink); }
.v-usage-count { font-variant-numeric: tabular-nums; font-weight: 700; }
.v-meter { height: 5px; border-radius: 3px; background: var(--line); overflow: hidden; }
.v-meter-fill { height: 100%; border-radius: 3px; background: var(--blue); transition: width 0.35s ease; }
.v-meter-fill.full { background: var(--rec); }
.v-pipeline { display: flex; flex-direction: column; }
.v-step { display: flex; align-items: center; gap: 0.65rem; padding: 0.42rem 0; font-size: 0.88rem; border-bottom: 1px solid var(--line); }
.v-step:last-child { border-bottom: none; }
.v-step-dot { width: 9px; height: 9px; flex: none; box-sizing: border-box; border-radius: 50%; border: 2px solid var(--line-2); }
.v-step-dot.active { border-color: var(--blue); background: var(--blue-soft); animation: v-blink 1.2s ease-in-out infinite; }
.v-step-dot.done { border-color: var(--ok); background: var(--ok); }
.v-step-dot.error { border-color: var(--bad); background: var(--bad); }
.v-step-label { flex: 1; font-weight: 500; }
.v-step.pending .v-step-label { color: var(--ink-3); }
.v-step-state { font-size: 0.78rem; color: var(--ink-3); }
.v-step.done .v-step-state { color: var(--ok); }
.v-step.active .v-step-state { color: var(--blue-ink); font-weight: 600; }
.v-step.error .v-step-state { color: var(--bad); font-weight: 600; }
.v-pipeline-error { margin-top: 0.7rem; padding: 0.65rem 0.8rem; border-left: 3px solid var(--bad); border-radius: 0 6px 6px 0; background: #FDF1F0; color: var(--bad); font-size: 0.84rem; line-height: 1.5; }
@keyframes v-blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
@keyframes v-march { from { background-position: 0 0; } to { background-position: 22.63px 0; } }
/* ── Hero ── */
.v-hero { max-width: 46rem; padding: 1.2rem 0 1.6rem; }
.v-hero-title { font-size: clamp(2.1rem, 4.4vw, 3.4rem); font-weight: 700; line-height: 1.07; letter-spacing: -0.028em; font-variation-settings: 'opsz' 96; color: var(--ink); }
.v-hero-sub { max-width: 38rem; margin-top: 1rem; font-family: var(--font-read); font-size: 1.18rem; line-height: 1.6; color: var(--ink-2); }
/* ── Composer: ruler edge ── */
.st-key-composer { padding: 1.1rem 1.4rem 1.3rem; background: var(--sheet); border: 1px solid var(--line-2); border-radius: 14px; box-shadow: 0 22px 40px -28px rgba(24, 32, 51, 0.35); gap: 1rem; }
.v-ruler { height: 22px; border: 1px solid var(--line); border-radius: 4px; background-color: var(--sheet-2); background-repeat: no-repeat; background-image: linear-gradient(var(--rec), var(--rec)), repeating-linear-gradient(90deg, var(--ink-3) 0 1px, transparent 1px 50px), repeating-linear-gradient(90deg, var(--line-2) 0 1px, transparent 1px 10px); background-size: 2px 100%, 100% 12px, 100% 7px; background-position: 12px 0, 0 100%, 0 100%; }
.v-composer-note { font-size: 0.84rem; color: var(--ink-3); }
[data-testid="stExpander"] details { background: var(--sheet); border: 1px solid var(--line-2); border-radius: 10px; }
[data-testid="stExpander"] summary { font-weight: 600; color: var(--ink); }
[data-testid="stExpander"] summary:hover { color: var(--blue); }
/* ── Inputs ── */
[data-testid="stWidgetLabel"] p { font-size: 0.84rem; font-weight: 600; color: var(--ink-2) !important; }
[data-testid="stTextInputRootElement"] { background-color: var(--sheet-2) !important; border: 1px solid var(--line-2) !important; border-radius: 8px !important; }
[data-testid="stTextInputRootElement"] div, [data-testid="stTextInputRootElement"] input { background-color: transparent !important; border: none !important; color: var(--ink) !important; -webkit-text-fill-color: var(--ink); }
[data-testid="stTextInputRootElement"]:focus-within { border-color: var(--blue) !important; box-shadow: 0 0 0 3px var(--blue-soft); }
[data-testid="stSelectbox"] [data-baseweb="select"] > div, [data-testid="stSelectbox"] [role="combobox"] { background-color: var(--sheet-2) !important; color: var(--ink) !important; border-color: var(--line-2) !important; }
[data-testid="stSelectbox"] [data-baseweb="select"] *, [data-testid="stSelectbox"] [role="combobox"] * { color: var(--ink) !important; -webkit-text-fill-color: var(--ink); }
[data-testid="stSelectbox"] svg { fill: var(--ink-3) !important; }
[role="listbox"], [role="listbox"] * { background-color: var(--sheet) !important; color: var(--ink) !important; }
[role="option"]:hover, [role="option"][aria-selected="true"] { background-color: var(--blue-soft) !important; }
[data-testid="stFileUploaderDropzone"] { background: var(--sheet-2); border: 1.5px dashed var(--line-2); border-radius: 10px; }
[data-testid="stFileUploaderDropzone"] span, [data-testid="stFileUploaderDropzone"] small { color: var(--ink-2) !important; }
[data-testid="stFileUploaderFile"] * { color: var(--ink) !important; }
/* Source switch: two option chips, the checked one is tinted */
[data-testid="stRadio"] [role="radiogroup"] { gap: 0.5rem 0.6rem; }
[data-testid="stRadio"] label { margin: 0; padding: 0.45rem 0.9rem; border: 1px solid var(--line-2); border-radius: 8px; background: var(--sheet-2); cursor: pointer; }
[data-testid="stRadio"] label:has(input:checked) { border-color: var(--blue); background: var(--blue-soft); }
[data-testid="stRadio"] label p, [data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] { color: var(--ink) !important; font-size: 0.92rem; font-weight: 600; }
/* ── Buttons ── */
.stButton > button { border-radius: 8px; font-weight: 600; transition: background-color 0.12s ease, border-color 0.12s ease, color 0.12s ease; }
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] { background: var(--blue); border: 1px solid var(--blue); color: #fff; padding: 0.6rem 1rem; }
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover { background: var(--blue-ink); border-color: var(--blue-ink); color: #fff; }
.stButton > button[kind="secondary"], [data-testid="stBaseButton-secondary"] { background: var(--sheet); border: 1px solid var(--line-2); color: var(--ink); }
.stButton > button[kind="secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover { border-color: var(--blue); color: var(--blue-ink); background: var(--blue-soft); }
/* ── Alerts ── */
[data-testid="stAlert"] { background: var(--sheet) !important; border: 1px solid var(--line-2); border-radius: 10px; }
[data-testid="stAlert"] p, [data-testid="stAlert"] div { color: var(--ink); }
[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) { border-left: 4px solid var(--bad); }
[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) { border-left: 4px solid var(--amber); }
[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) { border-left: 4px solid var(--ok); }
[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) { border-left: 4px solid var(--blue); }
/* ── Processing timeline ── */
.v-stage { margin: 1.5rem 0; padding: 1.1rem 1.3rem 1.3rem; background: var(--sheet); border: 1px solid var(--line-2); border-radius: 12px; }
.v-stage-head { margin-bottom: 1rem; font-size: 1.08rem; font-weight: 700; letter-spacing: -0.01em; }
.v-timeline { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 8px; }
.v-clip-bar { position: relative; height: 34px; display: flex; align-items: center; padding-left: 0.6rem; border: 1px solid var(--line-2); border-radius: 4px; background: var(--sheet-2); color: var(--ink-3); font-size: 0.82rem; font-weight: 700; }
.v-clip.done .v-clip-bar { background: var(--blue); border-color: var(--blue); color: #fff; }
.v-clip.active .v-clip-bar { border-color: var(--blue); color: var(--blue-ink); background-image: repeating-linear-gradient(135deg, var(--blue-soft) 0 8px, #fff 8px 16px); background-size: 22.63px 22.63px; animation: v-march 1s linear infinite; }
.v-clip.active .v-clip-bar::before { content: ""; position: absolute; left: -5px; top: -8px; bottom: -8px; width: 2px; border-radius: 1px; background: var(--rec); }
.v-clip.active .v-clip-bar::after { content: ""; position: absolute; left: -8px; top: -12px; width: 8px; height: 8px; border-radius: 2px; background: var(--rec); }
.v-clip.error .v-clip-bar { background: #FDECEC; border-color: var(--rec); color: var(--bad); }
.v-clip-name { margin-top: 0.55rem; font-size: 0.84rem; font-weight: 600; line-height: 1.25; }
.v-clip-state { font-size: 0.78rem; color: var(--ink-3); }
.v-clip.done .v-clip-state { color: var(--ok); }
.v-clip.active .v-clip-state { color: var(--blue-ink); font-weight: 600; }
.v-clip.error .v-clip-state { color: var(--bad); font-weight: 600; }
/* ── What you get ── */
.v-trio { display: grid; grid-template-columns: repeat(3, 1fr); margin-top: 2.2rem; }
.v-trio-item { padding: 0 1.5rem; border-left: 1px solid var(--line-2); }
.v-trio-item:first-child { padding-left: 0; border-left: none; }
.v-trio-title { margin-bottom: 0.3rem; font-size: 1.02rem; font-weight: 700; }
.v-trio-text { font-size: 0.93rem; line-height: 1.55; color: var(--ink-2); }
/* ── Result header ── */
.v-result-head { margin: 0.4rem 0 1.6rem; }
.v-result-title { max-width: 52rem; font-size: clamp(1.7rem, 3.2vw, 2.5rem); font-weight: 700; line-height: 1.14; letter-spacing: -0.022em; font-variation-settings: 'opsz' 80; }
.v-facts { display: flex; flex-wrap: wrap; row-gap: 0.8rem; margin-top: 1.1rem; }
.v-fact { display: flex; flex-direction: column; gap: 0.1rem; padding: 0 1.5rem; border-left: 1px solid var(--line-2); }
.v-fact:first-child { padding-left: 0; border-left: none; }
.v-fact-label { font-size: 0.8rem; color: var(--ink-3); }
.v-fact-value { font-size: 1rem; font-weight: 600; }
.v-fact-value.ok { color: var(--ok); }
.v-fact-value.ok::before { content: "✓ "; }
/* ── Tabs ── */
[role="tablist"] { gap: 1.8rem; border-bottom: 1px solid var(--line-2); }
button[role="tab"] { height: auto; padding: 0.6rem 0; background: transparent !important; }
button[role="tab"] p, button[role="tab"] div, button[role="tab"] span { color: var(--ink-3) !important; font-size: 1rem; font-weight: 600; }
button[role="tab"][aria-selected="true"] p, button[role="tab"][aria-selected="true"] div, button[role="tab"][aria-selected="true"] span { color: var(--ink) !important; }
button[role="tab"]:hover p { color: var(--blue-ink) !important; }
[role="tabpanel"] { padding-top: 1.6rem; }
/* ── Content ── */
.v-h2 { margin-bottom: 0.25rem; font-size: 1.25rem; font-weight: 700; letter-spacing: -0.01em; color: var(--ink); }
.v-lede { margin-bottom: 1rem; font-size: 0.95rem; color: var(--ink-2); }
.v-sheet { padding: 1.6rem 1.8rem; background: var(--sheet); border: 1px solid var(--line-2); border-radius: 12px; }
.v-read { max-width: 68ch; font-family: var(--font-read); font-size: 1.14rem; line-height: 1.78; color: var(--ink); overflow-wrap: anywhere; }
.v-guide-item { padding: 0.85rem 0; border-bottom: 1px solid var(--line-2); font-size: 0.93rem; line-height: 1.5; color: var(--ink-2); }
.v-guide-item:last-child { border-bottom: none; }
.v-guide-item b { display: block; color: var(--ink); font-size: 1rem; }
.v-list { height: 100%; padding: 1.2rem 1.4rem 1.4rem; background: var(--sheet); border: 1px solid var(--line-2); border-top: 4px solid var(--c); border-radius: 6px 6px 12px 12px; }
.v-list-action { --c: var(--blue); }
.v-list-decision { --c: var(--ok); }
.v-list-question { --c: var(--amber); }
.v-list-title { margin-bottom: 0.8rem; font-size: 1.1rem; font-weight: 700; }
.v-list-body { font-family: var(--font-read); font-size: 1.05rem; line-height: 1.7; overflow-wrap: anywhere; }
.v-empty { font-family: var(--font-ui); font-size: 0.92rem; color: var(--ink-3); }
.v-transcript { max-width: 76ch; max-height: 34rem; overflow-y: auto; padding: 1.5rem 1.8rem; background: var(--sheet); border: 1px solid var(--line-2); border-radius: 12px; font-family: var(--font-read); font-size: 1.06rem; line-height: 1.95; overflow-wrap: anywhere; }
/* ── Chat ── */
.v-chat-head { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 0.8rem; margin-bottom: 1rem; }
.v-quota { padding: 0.28rem 0.8rem; background: var(--sheet); border: 1px solid var(--line-2); border-radius: 999px; font-size: 0.84rem; font-weight: 600; color: var(--ink-2); }
.v-hint { margin-bottom: 0.6rem; font-size: 0.92rem; color: var(--ink-2); }
.st-key-suggestions .stButton > button { min-height: 4.4rem; height: auto; padding: 0.9rem 1rem; justify-content: flex-start; border-radius: 10px; }
.st-key-suggestions .stButton > button p { text-align: left; white-space: normal; font-size: 0.95rem; line-height: 1.35; }
[data-testid="stChatMessage"] { padding: 1rem 1.2rem; background: var(--sheet); border: 1px solid var(--line-2); border-radius: 12px; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: var(--blue-soft); border-color: transparent; }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { font-family: var(--font-read); font-size: 1.06rem; line-height: 1.7; color: var(--ink); }
[data-testid="stChatInput"] { background: var(--sheet); border: 1px solid var(--line-2); border-radius: 12px; }
[data-testid="stChatInput"] textarea { color: var(--ink); -webkit-text-fill-color: var(--ink); background: transparent; }
[data-testid="stSpinner"] * { color: var(--ink-2); }
.st-key-mobilebar { display: none; }
@media (max-width: 991px) {
  .st-key-mobilebar { display: flex; flex-direction: column; gap: 0.6rem; margin-bottom: 1.2rem; padding: 0.85rem 1rem; background: var(--sheet); border: 1px solid var(--line-2); border-radius: 12px; }
  [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"] { display: none !important; }
  .block-container { padding-top: 1.2rem; }
}
@media (max-width: 760px) {
  .v-timeline { grid-template-columns: repeat(3, minmax(0, 1fr)); row-gap: 1.1rem; }
  .v-trio { grid-template-columns: 1fr; row-gap: 1.2rem; }
  .v-trio-item { padding: 1rem 0 0; border-left: none; border-top: 1px solid var(--line-2); }
  .v-trio-item:first-child { padding-top: 0; border-top: none; }
  .v-fact { padding: 0 1rem; }
  .v-sheet, .v-transcript { padding: 1.2rem; }
}
@media (max-width: 480px) { .v-timeline { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (prefers-reduced-motion: reduce) { .v-clip.active .v-clip-bar, .v-step-dot.active { animation: none; } .v-meter-fill { transition: none; } }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "result": None,
    "chat_history": [],
    "processing": False,
    "pipeline_done": False,
    "pipeline_steps": {},
    "pipeline_error": None,
    # Language the completed analysis was actually run with (display only).
    "analysis_language": None,
}
# These counters intentionally live outside _DEFAULTS, so "+ New Analysis"
# cannot reset the usage limits.
_LIMIT_DEFAULTS = {
    "jobs_used": 0,
    "questions_used": 0,
}
for k, v in _LIMIT_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Per-browser-session Qdrant collection suffix. Not reset by "+ New Analysis".
if "rag_namespace" not in st.session_state:
    st.session_state.rag_namespace = uuid.uuid4().hex[:12]

for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _e(text: str) -> str:
    """Escape a short inline string for use inside custom HTML."""
    return html.escape(str(text))


def _e_block(text: str) -> str:
    """Escape AI-generated text for use inside a custom HTML block.

    Newlines become <br> so the markup stays on a single line. Raw newlines
    (especially blank lines or 4-space indents) would otherwise make Streamlit's
    markdown parser end the HTML block early and break the panel layout.
    """
    escaped = html.escape(str(text)).replace("\r\n", "\n").replace("\r", "\n")
    return escaped.replace("\n", "<br>")


def _list_body(text) -> str:
    """Escaped body for an insight list, with a plain note when it is empty."""
    if not str(text).strip():
        return '<span class="v-empty">Nothing was identified in this video.</span>'
    return _e_block(text)


def validate_video_duration(source: str) -> None:
    """Reject long remote videos before downloading their audio."""
    if not is_http_url(source):
        return

    assert_youtube_url(source)
    try:
        with yt_dlp.YoutubeDL(ydl_common_opts()) as ydl:
            info = ydl.extract_info(source, download=False)
    except UserFacingError:
        raise
    except Exception as exc:
        safe_log("Could not read video metadata", exc)
        raise UserFacingError(
            "We couldn't read this video URL. Check the link and try again."
        ) from exc

    if not isinstance(info, dict):
        raise UserFacingError(
            "The video's duration could not be verified, so it cannot be processed."
        )

    assert_single_youtube_video(info)
    duration = normalize_duration(info.get("duration"))
    assert_duration_within_limit(duration, MAX_VIDEO_MINUTES)


def save_uploaded_file(uploaded_file, work_dir: str) -> str:
    """Validate an upload and save it inside the job working directory."""
    name = getattr(uploaded_file, "name", "") or "upload"
    suffix = Path(name).suffix.lower()
    if suffix.lstrip(".") not in ALLOWED_EXTENSIONS:
        raise UserFacingError("This file type is not supported.")

    buffer = uploaded_file.getbuffer()
    actual_size = len(buffer)
    reported_size = getattr(uploaded_file, "size", None)
    assert_file_size(
        reported_size if reported_size is not None else actual_size,
        MAX_UPLOAD_MB,
    )
    assert_file_size(actual_size, MAX_UPLOAD_MB)
    assert_media_header(bytes(buffer[:16]))

    dest = os.path.join(work_dir, f"upload{suffix}")
    with open(dest, "wb") as handle:
        handle.write(buffer)
    return dest


PIPELINE_STEPS = [
    ("audio", "Audio processing"),
    ("transcript", "Transcription"),
    ("title", "Title generation"),
    ("summary", "Summarisation"),
    ("extract", "Extraction"),
    ("rag", "RAG initialisation"),
]

# Kept for compatibility with the original module; the redesigned UI
# communicates state with status dots, clip marks and text labels instead.
STEP_ICONS = {
    "audio": "◉",
    "transcript": "◈",
    "title": "◇",
    "summary": "◆",
    "extract": "◎",
    "rag": "◐",
}

# Placeholders so usage and pipeline status can refresh live, in place,
# while the pipeline is still running (no extra reruns required).
_SLOTS: dict = {}

_STEP_STATE_LABELS = {
    "pending": "Pending",
    "active": "Running",
    "done": "Done",
    "error": "Failed",
}


def render_usage_sidebar():
    """Render usage meters from the real session-state counters."""
    slots = [_SLOTS[n] for n in ("usage", "usage_m") if _SLOTS.get(n) is not None]
    if not slots:
        return

    def _meter(label: str, used: int, limit: int) -> str:
        pct = 0 if limit <= 0 else min(100, round(used / limit * 100))
        full = " full" if used >= limit else ""
        return (
            f'<div class="v-usage-row"><span>{label}</span>'
            f'<span class="v-usage-count">{used} / {limit}</span></div>'
            f'<div class="v-meter" role="progressbar" aria-label="{label} used" '
            f'aria-valuemin="0" aria-valuemax="{limit}" aria-valuenow="{used}">'
            f'<div class="v-meter-fill{full}" style="width:{pct}%"></div></div>'
        )

    usage_html = (
        '<div class="v-usage">'
        '<div class="v-usage-title">Usage this session</div>'
        + _meter("Analyses", int(st.session_state.jobs_used), MAX_JOBS_PER_SESSION)
        + _meter(
            "Questions", int(st.session_state.questions_used), MAX_QUESTIONS_PER_SESSION
        )
        + "</div>"
    )
    for slot in slots:
        slot.markdown(usage_html, unsafe_allow_html=True)


def render_stage_timeline():
    """Main-area processing timeline, driven by the same pipeline state."""
    slot = _SLOTS.get("stage")
    if slot is None:
        return

    steps = st.session_state.pipeline_steps
    if not steps or st.session_state.pipeline_done:
        slot.empty()
        return

    failed = False
    clips = []
    for index, (key, label) in enumerate(PIPELINE_STEPS, start=1):
        state = steps.get(key, "pending")
        if state not in _STEP_STATE_LABELS:
            state = "pending"
        failed = failed or state == "error"
        mark = "✓" if state == "done" else "!" if state == "error" else str(index)
        clips.append(
            f'<div class="v-clip {state}">'
            f'<div class="v-clip-bar"><span>{mark}</span></div>'
            f'<div class="v-clip-name">{label}</div>'
            f'<div class="v-clip-state">{_STEP_STATE_LABELS[state]}</div>'
            f"</div>"
        )

    heading = "Analysis stopped" if failed else "Analysing your video"
    slot.markdown(
        f'<div class="v-stage" role="status">'
        f'<div class="v-stage-head">{heading}</div>'
        f'<div class="v-timeline">{"".join(clips)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_pipeline_sidebar():
    """Render the six pipeline steps from the real pipeline state."""
    render_stage_timeline()

    slots = [_SLOTS[n] for n in ("pipeline", "pipeline_m") if _SLOTS.get(n) is not None]
    if not slots:
        return

    steps = st.session_state.pipeline_steps
    error = st.session_state.pipeline_error

    # Same visibility rule as before: only while there is progress to show.
    if not (steps or st.session_state.pipeline_done):
        for slot in slots:
            slot.empty()
        return

    rows = []
    for key, label in PIPELINE_STEPS:
        state = steps.get(key, "pending")
        if state not in _STEP_STATE_LABELS:
            state = "pending"
        rows.append(
            f'<div class="v-step {state}">'
            f'<span class="v-step-dot {state}" aria-hidden="true"></span>'
            f'<span class="v-step-label">{label}</span>'
            f'<span class="v-step-state">{_STEP_STATE_LABELS[state]}</span>'
            f"</div>"
        )

    error_html = (
        f'<div class="v-pipeline-error">{_e_block(error)}</div>' if error else ""
    )

    pipeline_html = (
        "<hr>"
        '<div class="v-sidebar-label" style="margin-top:0">Pipeline</div>'
        f'<div class="v-pipeline">{"".join(rows)}</div>'
        f"{error_html}"
    )
    for slot in slots:
        slot.markdown(pipeline_html, unsafe_allow_html=True)


def set_step(key: str, state: str):
    st.session_state.pipeline_steps[key] = state
    log.info("step=%s state=%s", key, state)
    render_pipeline_sidebar()


# Logo mark: viewfinder corners around a play triangle. The red dot (added in
# CSS) is the timeline playhead. Inline SVG, embedded as a data URI.
_LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40">'
    '<rect width="40" height="40" rx="10" fill="#3B4CE0"/>'
    '<g fill="none" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M9 15v-3a3 3 0 0 1 3-3h3"/><path d="M25 9h3a3 3 0 0 1 3 3v3"/>'
    '<path d="M31 25v3a3 3 0 0 1-3 3h-3"/><path d="M15 31h-3a3 3 0 0 1-3-3v-3"/></g>'
    '<path d="M17.5 15.5v9l7.5-4.5z" fill="#fff"/></svg>'
)
_LOGO_URI = "data:image/svg+xml;base64," + base64.b64encode(
    _LOGO_SVG.encode("utf-8")
).decode("ascii")

_WORDMARK_HTML = (
    '<div class="v-wordmark">'
    f'<span class="v-wordmark-icon"><img src="{_LOGO_URI}" alt="Framewise logo"></span>'
    "<div><div>Framewise</div>"
    '<div class="v-tagline">Video intelligence</div></div>'
    "</div>"
)


# ── Sidebar: identity, usage and pipeline status ─────────────────────────────────
with st.sidebar:
    # Wordmark
    st.markdown(_WORDMARK_HTML, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # New analysis button
    if st.button("+ New Analysis", use_container_width=True, type="secondary"):
        for k, v in _DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()

    # Usage meters (values always come from the session-state counters)
    _SLOTS["usage"] = st.empty()
    render_usage_sidebar()

    # Pipeline status — shown during and after processing
    _SLOTS["pipeline"] = st.empty()
    render_pipeline_sidebar()

    # Footer
    st.markdown(
        '<div class="v-footer">Framewise, version 1.0</div>', unsafe_allow_html=True
    )


# ── Compact header for small and medium screens (the sidebar is hidden there) ────
with st.container(key="mobilebar"):
    _bar_brand, _bar_new = st.columns([3, 2], gap="small", vertical_alignment="center")
    with _bar_brand:
        st.markdown(_WORDMARK_HTML, unsafe_allow_html=True)
    with _bar_new:
        if st.button(
            "+ New Analysis",
            use_container_width=True,
            type="secondary",
            key="new_analysis_top",
        ):
            for k, v in _DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()
    with st.expander("Usage and pipeline status"):
        _SLOTS["usage_m"] = st.empty()
        _SLOTS["pipeline_m"] = st.empty()
render_usage_sidebar()
render_pipeline_sidebar()


# ── Hero + composer (the input controls live in the main area) ───────────────────
_has_result = bool(st.session_state.result)

if not _has_result:
    st.markdown(
        '<div class="v-hero">'
        '<div class="v-hero-title">Find out what was said, decided, and left open.</div>'
        '<div class="v-hero-sub">Paste a YouTube link or upload a recording. Framewise transcribes it, '
        "writes the summary, pulls out the action items, and answers your follow-up questions.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

_composer_host = (
    st.expander("Analyse another video", expanded=False)
    if _has_result
    else st.container()
)

with _composer_host:
    with st.container(key="composer"):
        st.markdown('<div class="v-ruler"></div>', unsafe_allow_html=True)
        source_mode = st.radio(
            "Input type",
            ["YouTube URL / local path", "Upload file"],
            horizontal=True,
            label_visibility="collapsed",
            key="source_mode",
        )

        source = ""
        uploaded_file = None

        col_src, col_lang, col_run = st.columns(
            [5, 2, 2], gap="medium", vertical_alignment="bottom"
        )
        with col_src:
            if source_mode == "YouTube URL / local path":
                source = st.text_input(
                    "Source",
                    placeholder="Paste a YouTube URL or local file path",
                    label_visibility="collapsed",
                    key="source_input",
                )
            else:
                uploaded_file = st.file_uploader(
                    "Choose audio or video",
                    type=ALLOWED_EXTENSIONS,
                    max_upload_size=MAX_UPLOAD_MB,
                    help=f"Maximum file size: {MAX_UPLOAD_MB} MB.",
                    key="source_upload",
                )
        with col_lang:
            language = st.selectbox(
                "Language",
                ["english", "urdu"],
                index=0,
            )
        with col_run:
            run_btn = st.button(
                "Analyse video", use_container_width=True, type="primary"
            )

        st.markdown(
            f'<div class="v-composer-note">Videos up to {MAX_VIDEO_MINUTES} minutes, '
            f"uploads up to {MAX_UPLOAD_MB} MB, "
            f"{MAX_JOBS_PER_SESSION} analyses per browser session.</div>",
            unsafe_allow_html=True,
        )

# Processing timeline placeholder (filled live by set_step)
_SLOTS["stage"] = st.empty()
render_stage_timeline()


# ── Pipeline execution ────────────────────────────────────────────────────────────
if run_btn:
    if st.session_state.processing:
        st.warning("An analysis is already running. Please wait for it to finish.")
    elif st.session_state.jobs_used >= MAX_JOBS_PER_SESSION:
        st.error(
            "You've reached the video-analysis limit for this browser session. "
            "Please try again later."
        )
    elif source_mode == "YouTube URL / local path" and not source.strip():
        st.error("Enter a YouTube URL or local file path to continue.")
    elif source_mode == "Upload file" and uploaded_file is None:
        st.error("Choose an audio or video file to continue.")
    else:
        _t0 = time.perf_counter()
        status_slot = st.empty()
        work_dir = None
        processing_source = None

        try:
            if source_mode == "Upload file":
                name = getattr(uploaded_file, "name", "") or "upload"
                if Path(name).suffix.lower().lstrip(".") not in ALLOWED_EXTENSIONS:
                    raise UserFacingError("This file type is not supported.")
                assert_file_size(getattr(uploaded_file, "size", None), MAX_UPLOAD_MB)
                processing_source = "upload"
            else:
                processing_source = source.strip()
                if is_http_url(processing_source):
                    status_slot.info("Checking video duration…")
                    validate_video_duration(processing_source)
                else:
                    assert_safe_local_media(Path(processing_source), MAX_UPLOAD_MB)

            # Cheap validation passed. Paid work failures still consume a slot.
            st.session_state.jobs_used += 1
            render_usage_sidebar()
            log.info(
                "job start: mode=%s language=%s host=%s jobs_used=%s/%s",
                source_mode,
                language,
                (
                    urlparse(processing_source).netloc
                    if is_http_url(str(processing_source))
                    else "-"
                ),
                st.session_state.jobs_used,
                MAX_JOBS_PER_SESSION,
            )

            st.session_state.pipeline_done = False
            st.session_state.result = None
            st.session_state.analysis_language = None
            st.session_state.chat_history = []
            st.session_state.pipeline_steps = {}
            st.session_state.pipeline_error = None
            st.session_state.processing = True
            render_pipeline_sidebar()

            work_dir = tempfile.mkdtemp(prefix="vidora_job_")

            if source_mode == "Upload file":
                processing_source = save_uploaded_file(uploaded_file, work_dir)

            set_step("audio", "active")
            status_slot.info("Processing audio…")
            chunks = process_input(processing_source, work_dir)
            set_step("audio", "done")

            set_step("transcript", "active")
            status_slot.info("Transcribing…")
            transcript = transcribe_all(chunks, language)
            if not transcript or not transcript.strip():
                raise UserFacingError("No transcript was produced. Try another video.")
            set_step("transcript", "done")

            set_step("title", "active")
            status_slot.info("Generating title…")
            title = generate_title(transcript)
            set_step("title", "done")

            set_step("summary", "active")
            status_slot.info("Summarising…")
            summary = summarize(transcript)
            set_step("summary", "done")

            set_step("extract", "active")
            status_slot.info("Extracting insights…")
            action_items = extract_action_items(transcript)
            decisions = extract_key_decisions(transcript)
            questions = extract_questions(transcript)
            set_step("extract", "done")

            set_step("rag", "active")
            status_slot.info("Building knowledge index…")
            rag_collection = session_collection_name(st.session_state.rag_namespace)
            rag_chain = build_rag_chain(transcript, collection_name=rag_collection)
            set_step("rag", "done")

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
            }
            st.session_state.analysis_language = language
            st.session_state.pipeline_done = True
            status_slot.success("Analysis complete.")
            log.info("job complete")
            time.sleep(0.6)
            status_slot.empty()
            st.rerun()

        except UserFacingError as exc:
            _log_exc("Job rejected or failed", exc)
            for key, _ in PIPELINE_STEPS:
                if st.session_state.pipeline_steps.get(key) == "active":
                    set_step(key, "error")
            safe_message = str(exc)
            st.session_state.pipeline_error = safe_message
            render_pipeline_sidebar()
            status_slot.error(safe_message)

        except Exception as exc:
            _log_exc("Job crashed", exc)
            for key, _ in PIPELINE_STEPS:
                if st.session_state.pipeline_steps.get(key) == "active":
                    set_step(key, "error")
            safe_log("Processing failed", exc)
            safe_message = (
                "Analysis failed because of a processing or AI-service error. "
                "Please try again with a smaller video. If the problem continues, "
                "check the server logs."
            )
            st.session_state.pipeline_error = safe_message
            render_pipeline_sidebar()
            status_slot.error(safe_message)

        finally:
            log.info("job end: %.1fs", time.perf_counter() - _t0)
            st.session_state.processing = False
            if work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)


# ── What you get (empty state only) ──────────────────────────────────────────────
if not st.session_state.result:
    st.markdown(
        '<div class="v-trio">'
        '<div class="v-trio-item"><div class="v-trio-title">A summary you can read</div>'
        '<div class="v-trio-text">The main points in a few paragraphs, so you don\'t have to watch it again.</div></div>'
        '<div class="v-trio-item"><div class="v-trio-title">Actions, decisions, questions</div>'
        '<div class="v-trio-text">Each one pulled into its own list, ready to copy into your notes.</div></div>'
        '<div class="v-trio-item"><div class="v-trio-title">Answers from the transcript</div>'
        '<div class="v-trio-text">Ask follow-up questions and get replies based on what was actually said.</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )


# ── Results ───────────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    analysis_language = st.session_state.analysis_language or language

    _words = len(str(r["transcript"]).split())
    _read_min = max(1, round(_words / 200))

    # ── Result header ───────────────────────────────────────────────────────────
    st.markdown(
        '<div class="v-result-head">'
        f'<div class="v-result-title">{_e_block(r["title"])}</div>'
        '<div class="v-facts">'
        '<div class="v-fact"><span class="v-fact-label">Status</span>'
        '<span class="v-fact-value ok">Analysis complete</span></div>'
        '<div class="v-fact"><span class="v-fact-label">Transcript language</span>'
        f'<span class="v-fact-value">{_e(analysis_language.capitalize())}</span></div>'
        '<div class="v-fact"><span class="v-fact-label">Transcript length</span>'
        f'<span class="v-fact-value">{_words:,} words</span></div>'
        '<div class="v-fact"><span class="v-fact-label">Reading time</span>'
        f'<span class="v-fact-value">About {_read_min} min</span></div>'
        "</div></div>",
        unsafe_allow_html=True,
    )

    # ── Tabs ────────────────────────────────────────────────────────────────────
    tab_summary, tab_insights, tab_transcript, tab_chat = st.tabs(
        ["Summary", "Insights", "Transcript", "Ask AI"]
    )

    # ── Tab: Summary ────────────────────────────────────────────────────────────
    with tab_summary:
        col_main, col_rail = st.columns([3, 1.3], gap="large")
        with col_main:
            st.markdown(
                '<div class="v-h2">Executive summary</div>', unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="v-sheet"><div class="v-read">{_e_block(r["summary"])}</div></div>',
                unsafe_allow_html=True,
            )
        with col_rail:
            st.markdown(
                '<div class="v-h2">Where to look next</div>'
                '<div class="v-guide">'
                '<div class="v-guide-item"><b>Insights</b>Action items, key decisions, and open questions.</div>'
                '<div class="v-guide-item"><b>Transcript</b>The full text, to check any detail.</div>'
                '<div class="v-guide-item"><b>Ask AI</b>Questions answered from this transcript.</div>'
                "</div>",
                unsafe_allow_html=True,
            )

    # ── Tab: Insights ───────────────────────────────────────────────────────────
    with tab_insights:
        col_a, col_b, col_c = st.columns(3, gap="medium")

        with col_a:
            st.markdown(
                '<div class="v-list v-list-action"><div class="v-list-title">Action items</div>'
                f'<div class="v-list-body">{_list_body(r["action_items"])}</div></div>',
                unsafe_allow_html=True,
            )

        with col_b:
            st.markdown(
                '<div class="v-list v-list-decision"><div class="v-list-title">Key decisions</div>'
                f'<div class="v-list-body">{_list_body(r["key_decisions"])}</div></div>',
                unsafe_allow_html=True,
            )

        with col_c:
            st.markdown(
                '<div class="v-list v-list-question"><div class="v-list-title">Open questions</div>'
                f'<div class="v-list-body">{_list_body(r["open_questions"])}</div></div>',
                unsafe_allow_html=True,
            )

    # ── Tab: Transcript ─────────────────────────────────────────────────────────
    with tab_transcript:
        st.markdown(
            '<div class="v-h2">Full transcript</div>'
            f'<div class="v-lede">{_words:,} words. Scroll inside the page to read it all.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="v-transcript">{_e_block(r["transcript"])}</div>',
            unsafe_allow_html=True,
        )

    # ── Tab: Ask AI ─────────────────────────────────────────────────────────────
    with tab_chat:
        SUGGESTED = [
            "What were the key decisions made?",
            "Summarise the action items assigned.",
            "What open questions still need answers?",
        ]

        _left = max(0, MAX_QUESTIONS_PER_SESSION - int(st.session_state.questions_used))
        st.markdown(
            '<div class="v-chat-head"><div>'
            '<div class="v-h2">Ask anything about this video</div>'
            '<div class="v-lede" style="margin-bottom:0">Answers come from the transcript above.</div>'
            "</div>"
            f'<span class="v-quota">{_left} of {MAX_QUESTIONS_PER_SESSION} questions left</span>'
            "</div>",
            unsafe_allow_html=True,
        )

        if not st.session_state.chat_history:
            st.markdown(
                '<div class="v-hint">Suggested questions:</div>',
                unsafe_allow_html=True,
            )
            with st.container(key="suggestions"):
                sugg_cols = st.columns(len(SUGGESTED), gap="small")
                for sugg_col, q in zip(sugg_cols, SUGGESTED):
                    with sugg_col:
                        if st.button(
                            q,
                            key=f"sugg_{q}",
                            type="secondary",
                            use_container_width=True,
                        ):
                            if (
                                st.session_state.questions_used
                                >= MAX_QUESTIONS_PER_SESSION
                            ):
                                st.warning(
                                    "You've reached the question limit for this session."
                                )
                            else:
                                st.session_state.questions_used += 1
                                log.info(
                                    "question asked: %s/%s",
                                    st.session_state.questions_used,
                                    MAX_QUESTIONS_PER_SESSION,
                                )
                                try:
                                    with st.spinner("Thinking…"):
                                        answer = ask_question(r["rag_chain"], q)
                                except UserFacingError as exc:
                                    _log_exc("Question rejected or failed", exc)
                                    answer = str(exc)
                                except Exception as exc:
                                    _log_exc("Question crashed", exc)
                                    safe_log("Question failed", exc)
                                    answer = (
                                        "I couldn't get an answer from the AI service right now. "
                                        "Please try again later."
                                    )
                                st.session_state.chat_history.append(
                                    {"role": "user", "content": q}
                                )
                                st.session_state.chat_history.append(
                                    {"role": "assistant", "content": answer}
                                )
                                st.rerun()
        else:
            # Render history using native chat components
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            if st.button("Clear conversation", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

        # Chat input
        user_q = st.chat_input("Ask a question about this video…")
        if user_q and user_q.strip():
            if st.session_state.questions_used >= MAX_QUESTIONS_PER_SESSION:
                st.warning("You've reached the question limit for this session.")
            else:
                question = user_q.strip()
                st.session_state.questions_used += 1
                log.info(
                    "question asked: %s/%s",
                    st.session_state.questions_used,
                    MAX_QUESTIONS_PER_SESSION,
                )
                st.session_state.chat_history.append(
                    {"role": "user", "content": question}
                )
                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking…"):
                        try:
                            answer = ask_question(r["rag_chain"], question)
                        except UserFacingError as exc:
                            _log_exc("Question rejected or failed", exc)
                            answer = str(exc)
                        except Exception as exc:
                            _log_exc("Question crashed", exc)
                            safe_log("Question failed", exc)
                            answer = (
                                "I couldn't get an answer from the AI service right now. "
                                "Please try again later."
                            )
                    st.write(answer)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )
                st.rerun()
