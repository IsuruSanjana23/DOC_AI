from datetime import datetime

from pydantic import BaseModel, field_validator


class CollectionCreate(BaseModel):
    name: str
    description: str | None = None

    @field_validator("name")
    @classmethod
    def trim_and_validate_name(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Name must not be empty")
        if len(stripped) < 3:
            raise ValueError("Name must be at least 3 characters")
        if len(stripped) > 100:
            raise ValueError("Name must not exceed 100 characters")
        return stripped

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str | None) -> str | None:
        if v is not None:
            stripped = v.strip()
            if len(stripped) > 500:
                raise ValueError("Description must not exceed 500 characters")
            return stripped if stripped else None
        return v


class CollectionUpdate(BaseModel):
    name: str | None = None
    description: str | None = None

    @field_validator("name")
    @classmethod
    def trim_and_validate_name(cls, v: str | None) -> str | None:
        if v is not None:
            stripped = v.strip()
            if not stripped:
                raise ValueError("Name must not be empty")
            if len(stripped) < 3:
                raise ValueError("Name must be at least 3 characters")
            if len(stripped) > 100:
                raise ValueError("Name must not exceed 100 characters")
            return stripped
        return v

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str | None) -> str | None:
        if v is not None:
            stripped = v.strip()
            if len(stripped) > 500:
                raise ValueError("Description must not exceed 500 characters")
            return stripped if stripped else ""
        return v


class CollectionResponse(BaseModel):
    id: str
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
