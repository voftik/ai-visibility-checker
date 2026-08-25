from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "anthropic/claude-opus-5"
    OPENROUTER_ANALYSIS_MODEL: str = "anthropic/claude-opus-5"
    OPENROUTER_PROCESSING_MODEL: str = "openai/gpt-5.6-terra"
    OPENROUTER_CRITIC_MODEL: str = "google/gemini-3.6-flash"
    # Expensive decision layer. It is not part of the normal happy path and
    # may only choose from code-supplied, stage-specific recovery actions.
    OPENROUTER_ORCHESTRATOR_MODEL: str = "anthropic/claude-fable-5"
    # Expensive recovery is opt-in.  Production enables it explicitly only
    # after the deterministic canary path is healthy.
    PIPELINE_ORCHESTRATOR_ENABLED: bool = False
    PIPELINE_ORCHESTRATOR_MAX_CALLS_PER_RUN: int = 2
    # Deprecated compatibility knob. The long-response harness no longer
    # truncates orchestrator evidence by character count.
    PIPELINE_ORCHESTRATOR_MAX_INPUT_CHARS: int = 0
    OPENROUTER_ILLUSTRATION_CONCEPT_MODEL: str = "anthropic/claude-opus-5"
    OPENROUTER_IMAGE_MODEL: str = "google/gemini-3-pro-image"
    OPENROUTER_OPENAI_MODEL: str = "openai/gpt-chat-latest"
    OPENROUTER_GEMINI_MODEL: str = "google/gemini-3.6-flash"
    OPENROUTER_PERPLEXITY_MODEL: str = "perplexity/sonar-pro-search"
    OPENROUTER_DEEPSEEK_MODEL: str = "deepseek/deepseek-v4-pro"
    OPENROUTER_CLAUDE_MODEL: str = "anthropic/claude-sonnet-5"
    # Compatibility value for waiting on a concurrently claimed panel cell.
    OPENROUTER_TIMEOUT_SECONDS: int = 180
    # A generous inactivity deadline for one non-streaming provider POST.  This
    # is deliberately independent from output size/max tokens: it only keeps a
    # dead socket or wedged provider from holding a lease forever.
    OPENROUTER_READ_TIMEOUT_SECONDS: float = 7200.0
    OPENROUTER_PANEL_CONCURRENCY: int = 5
    # Deprecated compatibility knob. Kept so old deployment environments keep
    # parsing; the value is diagnostics-only and never gates or truncates input.
    FINAL_INPUT_TOKEN_BUDGET: int = 0
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEFAULT_CONCURRENCY: int = 8
    DEFAULT_TIMEOUT_SECONDS: int = 20
    # Normal audits use eight representative pages.  The crawler clamps any
    # per-run override to 6..10 and records the exact selected corpus in a
    # content-addressed manifest.  This bounds network work, not LLM output.
    AUDIT_PAGE_LIMIT: int = 8
    RUN_QUEUE_MAX_PENDING: int = 20
    RUN_LEASE_SECONDS: int = 90
    RUN_COORDINATOR_POLL_SECONDS: float = 3.0
    # Optional command for an isolated, unprivileged Playwright worker.
    # When empty, local development uses the bundled Python Playwright API.
    SITE_PREVIEW_WORKER_COMMAND: str = ""

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
