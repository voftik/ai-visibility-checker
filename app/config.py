from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "anthropic/claude-opus-4.7"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEFAULT_CONCURRENCY: int = 8
    DEFAULT_TIMEOUT_SECONDS: int = 15


settings = Settings()
