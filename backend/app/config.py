from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./mood_journal.db"
    emotion_model_name: str = "j-hartmann/emotion-english-distilroberta-base"
    enable_emotion_model: bool = True
    cors_origins: str = "http://localhost:3000,http://localhost:8080,http://127.0.0.1:8000"
    max_journal_characters: int = 10000
    bind_host: str = "127.0.0.1"
    enforce_loopback: bool = True
    auth_secret: str = "local-journal-session-secret"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="MOOD_")

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
