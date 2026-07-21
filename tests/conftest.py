"""Set required environment variables before any test module is imported.

Several application modules (``app.core.config``, ``app.models``) are
imported at the module level and trigger pydantic-settings validation.
Without these variables, ``Settings()`` raises a ``ValidationError``.
"""

import os

os.environ.setdefault("SECRET_KEY", "test-secret-key-not-for-production")
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://user:pass@localhost/test")
os.environ.setdefault("LLM_API_KEY", "sk-test-key-for-unit-tests")
