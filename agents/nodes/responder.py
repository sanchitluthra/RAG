import logfire
from agents.state import AgentState
from gateway import portkey_client, extract_cache_status


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    Uses the native Portkey client (not LangChain) so we can read the
    x-portkey-cache-status response header and surface Cache: Hit in the UI.
    """
    query = state["current_query"]

    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")
        prompt = f"""
        You are an Enterprise AI Assistant specializing in Kubernetes, Intel hardware, and enterprise networking.
        Respond to the user's message using the CONVERSATION HISTORY below.
        Keep the answer concise (under 250 words), focused, and clear.
        If the user's message is an off-topic question unrelated to Kubernetes, Intel hardware, networking, or previous conversation context (e.g., cooking, coffee, lifestyle, general trivia), politely state that you specialize in Enterprise IT infrastructure and decline to answer.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """
    else:
        logfire.info("Generating technical RAG response.")
        max_context_chars = 25000
        full_context = ""

        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break

        prompt = f"""
        You are a Senior Technical Architect.
        Answer the question using the TECHNICAL CONTEXT provided.
        Guidelines:
        - Keep your answer concise (at most ~250 words).
        - At most one comparison table if comparing concepts.
        - Avoid long cheat-sheets or repetitive lists; be direct and punchy.

        TECHNICAL CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """

    with logfire.span("✍️ LLM Synthesis"):
        for attempt in range(4):
            try:
                response = portkey_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                content = response.choices[0].message.content
                cache_status = extract_cache_status(response)
                is_cache_hit = cache_status == "HIT"

                if is_cache_hit:
                    logfire.info("⚡ Gateway Cache Hit — response served from Portkey cache.")
                    plan_update = state["plan"] + ["Cache: Hit ⚡"]
                    status = "Cache hit — instant response."
                else:
                    logfire.info("✅ Response synthesised via LLM.")
                    plan_update = state["plan"]
                    status = "Response generated."

                return {
                    "final_answer": content,
                    "status": status,
                    "plan": plan_update,
                    "messages": [{"role": "assistant", "content": content}]
                }

            except Exception as e:
                err_str = str(e).lower()
                if ("429" in err_str or "rate_limit" in err_str) and attempt < 3:
                    wait_time = (attempt + 1) * 5
                    logfire.warning(f"⚠️ Portkey/Groq Rate Limit (429) — retrying in {wait_time}s (attempt {attempt + 1}/3)...")
                    import time
                    time.sleep(wait_time)
                else:
                    logfire.error(f"LLM Generation failed: {e}")
                    raise e
