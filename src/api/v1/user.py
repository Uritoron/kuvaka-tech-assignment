# src/api/v1/user.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from src.schemas.chatroom import UserResponse
from src.models.user import User
from src.database.session import get_db
from src.core.security import get_current_user

router = APIRouter(prefix="/user", tags=["user"])

@router.get("/me", response_model=UserResponse)
def get_current_user_info(current_user: User = Depends(get_current_user)):
    return current_user