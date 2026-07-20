from functools import lru_cache

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

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()