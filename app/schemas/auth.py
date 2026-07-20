from pydantic import BaseModel, EmailStr


class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


class RegisterResponse(BaseModel):
    id: str
    name: str
    email: str

    model_config = {"from_attributes": True}


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    name: str
    email: str

    model_config = {"from_attributes": True}


class LogoutResponse(BaseModel):
    message: str = "Successfully logged out"
