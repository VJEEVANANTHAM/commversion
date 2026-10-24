"""Optional LLM adapter. The application remains usable without a key."""
from config import Settings

def make_llm():
    s = Settings()
    if not s.api_key:
        return None
    try:
        if s.provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=s.model, api_key=s.api_key, base_url=s.base_url or None, temperature=0)
        if s.provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=s.model, google_api_key=s.api_key, temperature=0)
        if s.provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(model=s.model, api_key=s.api_key, temperature=0)
        if s.provider == "openrouter":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=s.model, api_key=s.api_key, base_url=s.base_url or "https://openrouter.ai/api/v1", temperature=0)
        if s.provider == "mistral":
            from langchain_mistralai import ChatMistralAI
            return ChatMistralAI(model=s.model, api_key=s.api_key, temperature=0)
        if s.provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(model=s.model, temperature=0)
    except Exception as exc:
        print(f"[LLM unavailable; using deterministic fallback: {exc}]")
    return None
