import os
import re
from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv

from core.errors import UserFacingError, safe_log

load_dotenv()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
QDRANT_TIMEOUT_SECONDS = 15
_COLLECTION_SAFE = re.compile(r"[^a-zA-Z0-9_-]")


def _qdrant_url() -> str | None:
    return os.getenv("QDRANT_URL")


def _qdrant_api_key() -> str | None:
    return os.getenv("QDRANT_API_KEY")


def _collection_prefix() -> str:
    raw = os.getenv("QDRANT_COLLECTION", "video_transcripts") or "video_transcripts"
    cleaned = _COLLECTION_SAFE.sub("_", raw).strip("_")
    return cleaned or "video_transcripts"


def session_collection_name(namespace: str) -> str:
    token = _COLLECTION_SAFE.sub("", namespace)[:32]
    if not token:
        raise UserFacingError("Could not initialise the knowledge index.")
    return f"{_collection_prefix()}_{token}"


def _require_qdrant() -> tuple[str, str]:
    url = _qdrant_url()
    api_key = _qdrant_api_key()
    if not url or not api_key:
        raise UserFacingError(
            "The knowledge index is not configured. Please try again later."
        )
    return url, api_key


@lru_cache(maxsize=1)
def get_embeddings():
    safe_log("Loading embedding model")

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(
    transrcipt: str,
    collection_name: str | None = None,
) -> QdrantVectorStore:
    """Split transcript, generate embeddings, and store them in Qdrant."""

    safe_log("Building Qdrant vector store")

    if not transrcipt.strip():
        raise UserFacingError("Transcript cannot be empty.")

    url, api_key = _require_qdrant()
    name = collection_name or session_collection_name("default")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    chunks = splitter.split_text(transrcipt)

    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()

    vector_store = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        url=url,
        api_key=api_key,
        collection_name=name,
        timeout=QDRANT_TIMEOUT_SECONDS,
        force_recreate=True,
    )

    return vector_store


def load_vector_store(collection_name: str | None = None) -> QdrantVectorStore:
    """Connect to an existing Qdrant collection."""

    url, api_key = _require_qdrant()
    name = collection_name or session_collection_name("default")
    embeddings = get_embeddings()

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=url,
        api_key=api_key,
        collection_name=name,
        timeout=QDRANT_TIMEOUT_SECONDS,
    )

    return vector_store


def get_retriever(
    vector_store: QdrantVectorStore,
    k: int = 4,
):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
