import time
import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from service.config import settings

BATCH_SIZE = 50
GEMINI_DIM = 3072

_active_model = None


# ── Model initialisation ───────────────────────────────────────────────────────

def _init():
    """Initialise Gemini embedding model once per process."""
    global _active_model
    if _active_model is not None:
        return
    
    try:
        _active_model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=settings.GEMINI_API_KEY,
        )
        _active_model.embed_query("probe")
        logfire.info("Gemini embeddings ready (gemini-embedding-2-preview, 3072-dim).")
    except Exception as e:
        logfire.error(f"Gemini embedding init failed: {e}")
        raise


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_embedding_dim() -> int:
    _init()
    return GEMINI_DIM


# ── Batch embedding with retry ─────────────────────────────────────────────────
# 1 chunk 1 vector 
#rate limit error → wait and retry (max 4 times)
# any other error  → crash immediately, no retry
def _embed_batch(batch: list[str]) -> list[list[float]]:
    for attempt in range(4):
        try:
            # embed_document is sending to embedding model and then comnig bacl 
            return _active_model.embed_documents(batch)
        except Exception as e:
            err = str(e).lower()
            is_rate_limit = any(x in err for x in ("429", "rate", "quota", "resource_exhausted"))
            if is_rate_limit and attempt < 3:
                wait = 2 ** attempt
                logfire.warning(f"Gemini rate limit — retrying in {wait}s (attempt {attempt + 1}/4).")
                time.sleep(wait)
            else:
                logfire.error(f"Gemini embedding failed: {e}")
                raise
    raise RuntimeError("Gemini rate limit persisted after 4 attempts.")


# ── Public API ─────────────────────────────────────────────────────────────────

def embed_query(query: str) -> list[float]:
    _init()
    return _active_model.embed_query(query)


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        with logfire.span("Embed batch", start=i, size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))
    return all_embeddings