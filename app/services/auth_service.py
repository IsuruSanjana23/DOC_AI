from uuid import UUID

from sqlalchemy.orm import Session

from app.core.security import (
    create_access_token,
    hash_password,
    verify_password,
)
from app.repositories.user_repository import UserRepository
from app.schemas.auth import (
    LoginResponse,
    RegisterResponse,
    UserResponse,
)
from app.services.exceptions import (
    CredentialsException,
    DuplicateEmailException,
)


class AuthService:
    def __init__(self, db: Session):
        self.repo = UserRepository(db)

    def register(self, name: str, email: str, password: str) -> RegisterResponse:
        if self.repo.exists_by_email(email):
            raise DuplicateEmailException()

        password_hash = hash_password(password)
        user = self.repo.create(name=name, email=email, password_hash=password_hash)

        return RegisterResponse(
            id=str(user.id),
            name=user.name,
            email=user.email,
        )

    def login(self, email: str, password: str) -> LoginResponse:
        user = self.repo.get_by_email(email)
        if not user or not verify_password(password, user.password_hash):
            raise CredentialsException()

        access_token = create_access_token(subject=str(user.id))
        return LoginResponse(access_token=access_token)

    def get_current_user(self, user_id: UUID) -> UserResponse:
        user = self.repo.get_by_id(user_id)
        if not user:
            raise CredentialsException()
        return UserResponse(
            id=str(user.id),
            name=user.name,
            email=user.email,
        )
