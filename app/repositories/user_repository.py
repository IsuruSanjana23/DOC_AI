from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.user import User


class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, name: str, email: str, password_hash: str) -> User:
        user = User(name=name, email=email, password_hash=password_hash)
        self.db.add(user)
        self.db.flush()
        return user

    def get_by_id(self, user_id: UUID) -> User | None:
        return self.db.get(User, user_id)

    def get_by_email(self, email: str) -> User | None:
        stmt = select(User).where(User.email == email)
        return self.db.scalar(stmt)

    def exists_by_email(self, email: str) -> bool:
        stmt = select(User.id).where(User.email == email)
        return self.db.scalar(stmt) is not None
