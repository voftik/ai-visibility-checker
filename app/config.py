from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "anthropic/claude-opus-4.7"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEFAULT_CONCURRENCY: int = 8
    DEFAULT_TIMEOUT_SECONDS: int = 15

    # --- Outbound proxy pool (webshare.io) ---
    # Empty key disables proxying entirely; the crawler then talks directly,
    # which is the pre-existing behaviour.
    WEBSHARE_API_KEY: str = ""
    PROXY_ENABLED: bool = True
    PROXY_REFRESH_INTERVAL_SECONDS: int = 3600
    # When a proxied request errors out for connection/TLS reasons, retry the
    # same probe directly (no proxy). Keeps a single noisy proxy from poisoning
    # an entire run.
    PROXY_FALLBACK_DIRECT: bool = True
    PROXY_COOLDOWN_SECONDS: int = 300


settings = Settings()
