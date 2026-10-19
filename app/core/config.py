from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "marketmind-ollma-llm"
    app_env: str = "local"
    app_version: str = "0.1.0"

    log_level: str = "INFO"

    # Ollama runs on the host machine; from inside Docker it is reached via the
    # special host.docker.internal DNS name (see docker-compose extra_hosts).
    ollama_base_url: str = "http://host.docker.internal:11434"
    # Only a default: every request names its model, and this answers when one
    # does not.
    ollama_model: str = "qwen3:14b"
    # Local LLM inference on a 14B model is slow — generous timeout.
    ollama_timeout: int = 300
    ollama_num_ctx: int = 32768

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
