import hashlib
from typing import Optional

from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    page_content: str
    metadata: dict

    @property
    def room_number(self) -> str:
        return self.metadata.get('room_number')

    @property
    def description(self) -> str:
        return self.metadata.get('description')

    @property
    def room_size(self) -> float:
        return self.metadata.get('room_size')

    @property
    def image_url(self) -> Optional[str]:
        return self.metadata.get('image_url')

    @property
    def is_booked(self) -> bool:
        return self.metadata.get('is_booked', False)


class DocumentModel(BaseModel):
    room_number: str
    description: str
    room_size: float
    image_url: Optional[str] = None
    is_booked: bool = False
    page_content: str
    metadata: Optional[dict] = {}

    def generate_digest(self):
        hash_obj = hashlib.md5(self.page_content.encode())
        return hash_obj.hexdigest()
