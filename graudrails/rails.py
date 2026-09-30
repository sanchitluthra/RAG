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

groq_client: Groq | None = None

domain_embedding: np.ndarray | None = None

import re

# Strict whole-word cues: that, this, it, above, previous, earlier, simpler, shorter, summarize, example, elaborate
FOLLOW_UP_PATTERN = re.compile(
    r"\b(that|this|it|above|previous|earlier|simpler|shorter|summarize|example|elaborate)\b",
    re.IGNORECASE
)

# Common conversational and question words that do not represent distinct domain topics
COMMON_STOPWORDS = {
    "what", "is", "the", "difference", "between", "and", "or", "how", "to",
    "can", "you", "tell", "me", "about", "a", "an", "in", "of", "for", "with",
    "give", "show", "explain", "terms", "more", "less", "please", "why",
    "which", "who", "where", "when", "does", "do", "did", "are", "were", "was"
}


def extract_distinct_new_topic(current_query: str, history_text: str) -> bool:
    """
    Checks if current_query introduces a brand new distinct noun topic
    not present anywhere in the conversation history (e.g. 'photosynthesis', 'france').
    If a new topic word is found, Gate 4 must evaluate the current query alone.
    """

    if not history_text:
        return False

    hist_lower = history_text.lower()

    # Extract candidate topical words (alphanumeric words >= 4 chars not in common stopwords)
    words = re.findall(r"\b[a-zA-Z]{4,}\b", current_query.lower())

    for w in words:

        if FOLLOW_UP_PATTERN.search(w):
            continue

        if w in COMMON_STOPWORDS:
            continue

        # If this meaningful word never appeared in prior history, it is a new topic!
        if not re.search(r"\b" + re.escape(w) + r"\b", hist_lower):
            return True

    return False


def get_similarity(text: str) -> float:
    """Computes cosine similarity of text against domain embedding."""

    query_vec = np.array(embed_query(text))

    similarity = np.dot(query_vec, domain_embedding) / (
        np.linalg.norm(query_vec) * np.linalg.norm(domain_embedding)
    )

    return float(similarity)


def initialize_rails() -> None:
    """Initialise Groq client and cache the domain embedding (called once at startup)."""

    global groq_client, domain_embedding

    groq_client = Groq(api_key=settings.GROQ_API_KEY)

    domain_embedding = np.array(embed_query(DOMAIN_DESCRIPTION))

    logfire.info("🛡️ Guardrails initialised.")


def guard(
    message: str,
    has_history: bool = False,
    last_user_question: str = "",
    history_text: str = ""
) -> tuple[bool, str | None]:

    """
    Multi-gate guardrail:

    1. Dialog keywords
    2. Off-topic keywords
    3. LlamaPromptGuard (always runs on every message)
    4. Domain Similarity Check (0.55 threshold):
       - If has_history and whole-word cue matches AND no new distinct topic introduced:
         embeds '<last user question> + <current query>'
       - Otherwise: embeds current query alone
    """

    if groq_client is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("🛡️ Guardrails Check"):

        msg_lower = message.strip().lower()

        # Gate 1: Dialog keywords

        if any(trigger in msg_lower for trigger in GREETING_TRIGGERS):
            logfire.info("🛡️ Gate 1 dialog response | dialog=greeting", gate=1)
            return True, GREETING_RESPONSE

        if any(trigger in msg_lower for trigger in FAREWELL_TRIGGERS):
            logfire.info("🛡️ Gate 1 dialog response | dialog=farewell", gate=1)
            return True, FAREWELL_RESPONSE

        if any(trigger in msg_lower for trigger in CAPABILITIES_TRIGGERS):
            logfire.info("🛡️ Gate 1 dialog response | dialog=capabilities", gate=1)
            return True, CAPABILITIES_RESPONSE

        # Gate 2: Off-topic keywords

        for keyword in OFF_TOPIC_KEYWORDS:

            if keyword in msg_lower:
                logfire.info(
                    f"🛡️ Gate 2 blocked | off-topic keyword='{keyword}'",
                    gate=2,
                    matched_keyword=keyword
                )
                return True, OFF_TOPIC_RESPONSE

        # Gate 3: LlamaPromptGuard — jailbreak/injection (ALWAYS runs on every message)

        try:

            response = groq_client.chat.completions.create(
                model="meta-llama/llama-prompt-guard-2-86m",
                messages=[{"role": "user", "content": message}]
            )

            label = response.choices[0].message.content.strip()

            logfire.info(
                f"🛡️ PromptGuard label: '{label}'",
                gate=3
            )

            score = float(label)

            if score > 0.5:
                logfire.info(
                    f"🛡️ Gate 3 blocked | jailbreak score={score:.4f}",
                    gate=3,
                    jailbreak_score=round(score, 4)
                )
                return True, JAILBREAK_RESPONSE

        except ValueError:

            label_upper = label.upper()

            if (
                "INJECTION" in label_upper
                or "JAILBREAK" in label_upper
                or "UNSAFE" in label_upper
            ):
                logfire.info(
                    f"🛡️ Gate 3 blocked | jailbreak label={label}",
                    gate=3,
                    jailbreak_label=label
                )
                return True, JAILBREAK_RESPONSE

        except Exception as e:
            logfire.warning(f"⚠️ PromptGuard check failed: {e}")

        # Gate 4: Domain Check (Cosine Similarity >= 0.55)

        # Determine text to evaluate

        has_cue = bool(FOLLOW_UP_PATTERN.search(message))

        has_new_topic = extract_distinct_new_topic(
            message,
            history_text or last_user_question
        )

        if has_history and has_cue and not has_new_topic:

            eval_text = (
                f"{last_user_question} + {message}"
                if last_user_question
                else message
            )

            eval_mode = "combined"

        else:

            eval_text = message
            eval_mode = "standalone"

        sim = get_similarity(eval_text)

        logfire.info(
            f"🛡️ Gate 4 similarity: {sim:.4f} ({eval_mode})",
            gate=4,
            similarity_score=round(sim, 4),
            eval_mode=eval_mode
        )

        if sim < 0.55:

            logfire.info(
                "🛡️ Gate 4 blocked | off-topic semantic",
                gate=4,
                similarity_score=round(sim, 4)
            )

            return True, OFF_TOPIC_RESPONSE

        logfire.info("✅ Guardrails passed.")

        return False, None
