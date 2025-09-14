# src/models/chatroom.py
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, func ,Boolean 
from sqlalchemy.orm import relationship
from src.database.base import Base

class Chatroom(Base):
    __tablename__ = "chatrooms"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=func.now())

    owner = relationship("User", back_populates="chatrooms")  # Must match User.chatrooms
    messages = relationship("Message", back_populates="chatroom", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(String, nullable=False)
    is_from_user = Column(Boolean, default=True)
    chatroom_id = Column(Integer, ForeignKey("chatrooms.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=func.now())

    chatroom = relationship("Chatroom", back_populates="messages")
    user = relationship("User", back_populates="messages")