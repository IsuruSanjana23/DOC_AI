from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "DocCollection AI"
    app_version: str = "1.0.0"
    debug: bool = False

    api_v1_prefix: str = "/api/v1"

    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    database_url: str

    upload_dir: str = "uploads"
    max_file_size_mb: int = 20

    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"

    llm_api_key: str = Field(default="", validation_alias="LITELLM_API_KEY")
    llm_base_url: str = "https://lite-llm.datafabdevelopment.com/v1"
    llm_model: str = "deepseek-v4-flash"
    llm_timeout_seconds: int = 60
    llm_max_retries: int = 3

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()