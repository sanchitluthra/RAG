import logfire
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI

from service.config import settings


# Production gateway config:
#   - Fallback: primary @rag//openai/gpt-oss-120b → @brag//openai/gpt-oss-20b on failure
#   - Cache: semantic mode (requires Portkey Enterprise — silently falls back to simple on free/starter)
#   - Retry: 2 attempts on rate limit / server error before triggering the fallback target
CONFIG_SLUG = "pc-rag-1aea48"


portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY,
    config=CONFIG_SLUG
)


def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI — a drop-in for ChatGroq in LangChain nodes.

    Why ChatOpenAI and not ChatGroq:
      Portkey is a proxy. It exposes an OpenAI-compatible endpoint at PORTKEY_GATEWAY_URL.
      support routing through a proxy.
      ChatOpenAI supports base_url (points at Portkey) and default_headers (passes Portkey
      auth + config). The @rag/model-name format is Portkey-specific — Groq's own client
      does not understand it. You are still using Groq models; Portkey is just in the middle.
    """
    return ChatOpenAI(
    api_key=settings.PORTKEY_API_KEY,
    base_url=PORTKEY_GATEWAY_URL,
    model="openai/gpt-oss-120b",
    temperature=0,
    default_headers=createHeaders(
        api_key=settings.PORTKEY_API_KEY,
        config="pc-rag-1aea48",
        metadata={
            "feature": feature,
            "_user": "rag-system",
            "environment": "production"
        }
    )
)

def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey native client response headers.
    Tries multiple attribute paths defensively — returns 'MISS' if not found.
    """
    # 1. Direct headers or _headers dict on response
    headers = getattr(response, "headers", None) or getattr(response, "_headers", None)
    if headers:
        if hasattr(headers, "get"):
            status = headers.get("x-portkey-cache-status", "")
            if status:
                return str(status).upper()

    # 2. Check underlying raw/http responses
    for attr in ("_raw_response", "_response", "_http_response", "raw_response", "http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            raw_headers = getattr(raw, "headers", {})
            if hasattr(raw_headers, "get"):
                status = raw_headers.get("x-portkey-cache-status", "")
                if status:
                    return status.upper()

    # 3. Check choices or metadata if Portkey attached headers there
    meta = getattr(response, "_metadata", None) or getattr(response, "metadata", None)
    if meta and hasattr(meta, "get"):
        status = meta.get("x-portkey-cache-status", "")
        if status:
            return status.upper()

    return "MISS"
