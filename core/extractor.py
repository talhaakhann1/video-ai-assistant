from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
import os

from core.errors import UserFacingError, classify_upstream_error, safe_log


def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise UserFacingError(
            "The AI service is not configured. Please try again later."
        )
    return ChatOpenAI(
        model="openai/gpt-oss-20b",
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        temperature=0.2,
        timeout=60,
        max_retries=2,
    )


def build_chain(system_prompt: str):

    if not system_prompt.strip():
        raise UserFacingError("system_prompt is needed.")

    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}"),
    ]) | llm | StrOutputParser()
    )


def _invoke_chain(chain, transcript: str) -> str:
    try:
        return chain.invoke(transcript)
    except Exception as exc:
        mapped = classify_upstream_error(exc)
        if mapped:
            raise mapped from exc
        safe_log("Extraction request failed", exc)
        raise UserFacingError(
            "The AI service could not extract insights. Please try again."
        ) from exc


def extract_key_decisions(transcript: str) -> str:

    if not transcript.strip():
        raise UserFacingError("Transcript cannot be empty.")

    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    return _invoke_chain(chain, transcript)


def extract_action_items(transcript: str) -> str:

    if not transcript.strip():
        raise UserFacingError("Transcript cannot be empty.")

    chain = build_chain(
         "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )

    return _invoke_chain(chain, transcript)


def extract_questions(transcript: str) -> str:

    if not transcript.strip():
        raise UserFacingError("Transcript cannot be empty.")

    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    return _invoke_chain(chain, transcript)
