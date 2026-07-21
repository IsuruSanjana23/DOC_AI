from datetime import datetime

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: str
    original_filename: str
    mime_type: str
    file_size: int
    status: str
    collection_id: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
