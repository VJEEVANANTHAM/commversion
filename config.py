"""Environment-backed configuration."""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    provider: str = os.getenv("LLM_PROVIDER", "groq").lower()
    model: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    api_key: str = os.getenv("LLM_API_KEY", "")
    base_url: str = os.getenv("LLM_BASE_URL", "")
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "")
    data_dir: str = os.getenv("DATA_DIR", "data")
