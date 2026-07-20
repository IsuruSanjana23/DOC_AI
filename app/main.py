from fastapi import FastAPI

from app.api.v1.auth import router as auth_router
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)

app.include_router(auth_router, prefix=settings.api_v1_prefix)


@app.get("/")
def root():
    return {
        "message": settings.app_name,
        "version": settings.app_version,
    }
    
from sqlalchemy import text

from app.db.session import engine


@app.get("/db-check")
def db_check():
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))

    return {"database": result.scalar()}