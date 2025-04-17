import hashlib
from typing import Optional

from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    room_name: str
    room_number: str
    description: str
    room_size: float
    image_url: Optional[str] = None
    is_booked: bool = False
    metadata: Optional[dict] = {}

class DocumentModel(BaseModel):
    room_number: str
    description: str
    room_size: float
    image_url: Optional[str] = None
    is_booked: bool = False
    room_name: str
    metadata: Optional[dict] = {}
    
    def generate_digest(self):
        hash_obj = hashlib.md5(self.room_name.encode())
        return hash_obj.hexdigest()
