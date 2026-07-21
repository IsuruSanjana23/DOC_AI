"""Set required environment variables for service-layer tests.

Without these, pydantic-settings raises a ValidationError when
importing ``app.core.config.settings`` because several fields
(secret_key, database_url, llm_api_key) have no defaults.
"""

import os

os.environ.setdefault("SECRET_KEY", "test-secret-key-not-for-production")
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://user:pass@localhost/test")
os.environ.setdefault("LITELLM_API_KEY", "sk-test-key-for-unit-tests")
