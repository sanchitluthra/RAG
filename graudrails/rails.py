import logfire
from groq import Groq

from service.config import settings
from graudrails.colang_rules import (
    GREETING_TRIGGERS, FAREWELL_TRIGGERS, CAPABILITIES_TRIGGERS,
    GREETING_RESPONSE, FAREWELL_RESPONSE, CAPABILITIES_RESPONSE,
    OFF_TOPIC_RESPONSE, JAILBREAK_RESPONSE, OFF_TOPIC_KEYWORDS
)

_groq_client: Groq | None = None


def initialize_rails() -> None:
    """
    Initialize Groq client for llama-prompt-guard-2-86m.
    This model is a fine-tuned classifier for jailbreak and prompt injection detection.
    Dialog (greetings, farewell, capabilities) handled via fast keyword matching.
    """
    global _groq_client
    _groq_client = Groq(api_key=settings.GROQ_API_KEY)
    logfire.info("🛡️ Guardrails initialised (llama-prompt-guard-2-86m + keyword matching).")


def guard(message: str) -> tuple[bool, str | None]:
    """
    Gate 1: Fast keyword matching for dialog (no LLM call).
    Gate 2: llama-prompt-guard-2-86m for jailbreak/injection detection.
    Gate 3: Off-topic keyword matching.
    Returns (fired, response).
    """
    if _groq_client is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("🛡️ Guardrails Check"):
        msg_lower = message.strip().lower()

        # Gate 1: Dialog — pure keyword match, zero LLM calls
        if any(trigger in msg_lower for trigger in GREETING_TRIGGERS):
            logfire.info("🛡️ Guardrails fired | dialog=greeting")
            return True, GREETING_RESPONSE

        if any(trigger in msg_lower for trigger in FAREWELL_TRIGGERS):
            logfire.info("🛡️ Guardrails fired | dialog=farewell")
            return True, FAREWELL_RESPONSE

        if any(trigger in msg_lower for trigger in CAPABILITIES_TRIGGERS):
            logfire.info("🛡️ Guardrails fired | dialog=capabilities")
            return True, CAPABILITIES_RESPONSE

        # Gate 2: llama-prompt-guard for jailbreak/injection detection
        # Returns a probability score (0.0 to 1.0) — higher = more dangerous
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
            # Not a float — might be a text label like UNSAFE
            label_upper = label.upper()
            if "INJECTION" in label_upper or "JAILBREAK" in label_upper or "UNSAFE" in label_upper:
                logfire.info(f"🛡️ Guardrails fired | jailbreak label={label}")
                return True, JAILBREAK_RESPONSE
        except Exception as e:
            logfire.warning(f"⚠️ PromptGuard check failed: {e}")

        # Gate 3: Off-topic keyword matching
        if any(keyword in msg_lower for keyword in OFF_TOPIC_KEYWORDS):
            logfire.info(f"🛡️ Guardrails fired | off-topic detected")
            return True, OFF_TOPIC_RESPONSE

        logfire.info("✅ Guardrails passed.")
        return False, None