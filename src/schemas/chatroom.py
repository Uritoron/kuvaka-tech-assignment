# src/schemas/chatroom.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional

class ChatroomCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)

class ChatroomResponse(BaseModel):
    id: int
    name: str
    created_at: datetime

    class Config:
        from_attributes = True

class MessageCreate(BaseModel):
    content: str = Field(..., min_length=1)

class MessageResponse(BaseModel):
    id: int
    content: str
    is_from_user: bool
    created_at: datetime

    class Config:
        from_attributes = True

class UserResponse(BaseModel):
    id: int
    mobile_number: str
    subscription_tier: str
    daily_message_count: int
    last_message_date: datetime
    created_at: datetime

    class Config:
        from_attributes = True