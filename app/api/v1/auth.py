from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.dependencies.auth import get_current_user, get_db
from app.schemas.auth import (
    LoginRequest,
    LoginResponse,
    LogoutResponse,
    RegisterRequest,
    RegisterResponse,
    UserResponse,
)
from app.services.auth_service import AuthService
from app.services.exceptions import CredentialsException, DuplicateEmailException

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    service = AuthService(db)
    try:
        return service.register(
            name=body.name,
            email=body.email,
            password=body.password,
        )
    except DuplicateEmailException:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email already exists",
        )


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    service = AuthService(db)
    try:
        return service.login(email=body.email, password=body.password)
    except CredentialsException:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/logout", response_model=LogoutResponse)
def logout():
    return LogoutResponse()


@router.get("/me", response_model=UserResponse)
def me(current_user: UserResponse = Depends(get_current_user)):
    return current_user
