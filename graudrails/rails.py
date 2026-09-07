import logfire
import numpy as np
from groq import Groq

from service.config import settings
from service.retrival.embedding import embed_query
from graudrails.colang_rules import (
    GREETING_TRIGGERS, FAREWELL_TRIGGERS, CAPABILITIES_TRIGGERS,
    GREETING_RESPONSE, FAREWELL_RESPONSE, CAPABILITIES_RESPONSE,
    OFF_TOPIC_RESPONSE, JAILBREAK_RESPONSE, OFF_TOPIC_KEYWORDS,
    DOMAIN_DESCRIPTION
)

_groq_client: Groq | None = None
_domain_embedding: np.ndarray | None = None


def initialize_rails() -> None:
    """
    Gate 1: Dialog keywords (instant)
    Gate 2: Off-topic keywords (instant)
    Gate 3: LlamaPromptGuard (jailbreak/injection)
    Gate 4: Semantic similarity (domain check)
    """
    global _groq_client, _domain_embedding

    _groq_client = Groq(api_key=settings.GROQ_API_KEY)
    _domain_embedding = np.array(embed_query(DOMAIN_DESCRIPTION))

    logfire.info("🛡️ Guardrails initialised (keyword + llama-prompt-guard + semantic similarity).")


def _is_off_topic(message: str) -> bool:
    """
    Gate 4 — Semantic similarity using Gemini embeddings.
    Same vector space as RAG = accurate domain filtering.
    Threshold 0.55 — tune based on testing.
    """
    query_vec = np.array(embed_query(message))
    similarity = np.dot(query_vec, _domain_embedding) / (
        np.linalg.norm(query_vec) * np.linalg.norm(_domain_embedding)
    )
    logfire.info(f"🛡️ Semantic similarity score: {similarity:.4f}")
    return float(similarity) < 0.62


def guard(message: str) -> tuple[bool, str | None]:
    if _groq_client is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("🛡️ Guardrails Check"):
        msg_lower = message.strip().lower()

        # Gate 1: Dialog keywords
        if any(trigger in msg_lower for trigger in GREETING_TRIGGERS):
            logfire.info("🛡️ Guardrails fired | dialog=greeting")
            return True, GREETING_RESPONSE

        if any(trigger in msg_lower for trigger in FAREWELL_TRIGGERS):
            logfire.info("🛡️ Guardrails fired | dialog=farewell")
            return True, FAREWELL_RESPONSE

        if any(trigger in msg_lower for trigger in CAPABILITIES_TRIGGERS):
            logfire.info("🛡️ Guardrails fired | dialog=capabilities")
            return True, CAPABILITIES_RESPONSE

        # Gate 2: Off-topic keywords
        if any(keyword in msg_lower for keyword in OFF_TOPIC_KEYWORDS):
            logfire.info("🛡️ Guardrails fired | off-topic keyword")
            return True, OFF_TOPIC_RESPONSE

        # Gate 3: LlamaPromptGuard — jailbreak/injection
        try:
            response = _groq_client.chat.completions.create(
                model="meta-llama/llama-prompt-guard-2-86m",
                messages=[{"role": "user", "content": message}]
            )
            label = response.choices[0].message.content.strip()
            logfire.info(f"🛡️ PromptGuard label: '{label}'")

            score = float(label)
            if score > 0.5:
                logfire.info(f"🛡️ Guardrails fired | jailbreak score={score:.4f}")
                return True, JAILBREAK_RESPONSE

        except ValueError:
            label_upper = label.upper()
            if "INJECTION" in label_upper or "JAILBREAK" in label_upper or "UNSAFE" in label_upper:
                logfire.info(f"🛡️ Guardrails fired | jailbreak label={label}")
                return True, JAILBREAK_RESPONSE
        except Exception as e:
            logfire.warning(f"⚠️ PromptGuard check failed: {e}")

        # Gate 4: Semantic similarity — domain check
        if _is_off_topic(message):
            logfire.info("🛡️ Guardrails fired | off-topic semantic")
            return True, OFF_TOPIC_RESPONSE

        logfire.info("✅ Guardrails passed.")
        return False, None
