# src/models/user.py
from sqlalchemy import Column, Integer, String, Boolean, DateTime, func
from sqlalchemy.orm import relationship
from src.database.base import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    mobile_number = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    subscription_tier = Column(String, default="Basic")
    daily_message_count = Column(Integer, default=0)
    last_message_date = Column(DateTime, default=func.now())
    created_at = Column(DateTime, default=func.now())

    # 👇 This must be here — and match the back_populates in Chatroom
    chatrooms = relationship("Chatroom", back_populates="owner", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="user", cascade="all, delete-orphan")