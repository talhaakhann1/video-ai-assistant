from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

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
        temperature=0.3,
        timeout=60,
        max_retries=2,
    )


def split_transcript(transcript: str) -> str:

    if not transcript.strip():
        raise UserFacingError("Transcript cannot be empty.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )

    return splitter.split_text(transcript)


def _invoke_llm(chain, payload):
    try:
        return chain.invoke(payload)
    except Exception as exc:
        mapped = classify_upstream_error(exc)
        if mapped:
            raise mapped from exc
        safe_log("Summarization request failed", exc)
        raise UserFacingError(
            "The AI service could not finish summarising. Please try again."
        ) from exc


def summarize(transcript: str) -> str:

    if not transcript.strip():
        raise UserFacingError("Transcript cannot be empty.")

    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", "Summarize this portion of a meeting transcript concisely."),
        ("human", "{text}"),
    ]
    )

    chain_prompt = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    chunk_summaries = [_invoke_llm(chain_prompt, {"text": chunk}) for chunk in chunks]

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
        (
            "system",
            "You are an expert meeting summarizer. Combine these partial summaries "
            "into one final professional meeting summary in bullet points.",
        ),
        ("human", "{text}"),
    ]
    )

    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | combined_prompt | llm | StrOutputParser()
    )

    return _invoke_llm(combined_chain, combined)


def generate_title(transcript: str) -> str:

    if not transcript.strip():
        raise UserFacingError("Transcript cannot be empty.")

    llm = get_llm()

    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) |
        ChatPromptTemplate.from_messages([
             (
                "system",
                "Based on the meeting transcript, generate a short professional meeting title "
                "(max 8 words). Only return the title, nothing else.",
            ),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )

    return _invoke_llm(title_chain, transcript[:2000])
