# Colang and YAML no longer needed for llama-prompt-guard
# Keeping only the response messages and block keywords

GREETING_TRIGGERS = ["hello", "hi", "hey", "good morning", "good afternoon", "what's up", "howdy"]
FAREWELL_TRIGGERS = ["bye", "goodbye", "see you", "thanks bye", "that is all", "i am done", "see you later"]
CAPABILITIES_TRIGGERS = ["what can you do", "what do you know", "what are you", "what topics do you cover", "what can i ask you", "what are your capabilities""who are you",        # ADD THIS
    "tell me about yourself", 
    "introduce yourself" ]      

GREETING_RESPONSE = "Hello! I'm your Enterprise IT Assistant. I specialise in Kubernetes, Intel hardware, and enterprise networking. What can I help you with today?"
FAREWELL_RESPONSE = "Goodbye! Feel free to return whenever you have more enterprise IT questions. Have a great day!"
CAPABILITIES_RESPONSE = "I'm an Enterprise AI Assistant with deep expertise in: Kubernetes (deployment, scaling, networking, operators), Intel Hardware (CPUs, FPGAs, SRIOV, NICs), Enterprise Networking (SDN, VLANs, BGP, routing). Ask me anything in these areas!"
OFF_TOPIC_RESPONSE = "I'm an Enterprise IT Assistant focused on Kubernetes, Intel hardware, and networking. I can't help with that — but ask me anything technical!"
JAILBREAK_RESPONSE = "I maintain consistent guidelines regardless of how I am prompted. I am here to help with Kubernetes, Intel, and networking. What can I help you with?"

# Off-topic keywords — if message contains these, block immediately
OFF_TOPIC_KEYWORDS = [
    "joke", "poem", "weather", "recipe", "cook", "coffee",
    "movie", "song", "president", "history", "restaurant",
    "homework", "math", "stock", "invest", "story", "capital of",
    "who won", "recommend a", "tell me about", "what should i eat",
    "dinner", "lunch", "breakfast", "food", "bake", "cake"
]