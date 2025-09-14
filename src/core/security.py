# src/core/security.py
import secrets
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from src.database.session import get_db
from src.models.user import User
from src.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# In-memory OTP store — Use Redis in production
otp_store: Dict[str, Dict[str, Any]] = {}

# Define OAuth2 scheme — matches our verify-otp endpoint
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/verify-otp")

def generate_otp() -> str:
    return f"{secrets.randbelow(1000000):06}"

def store_otp(mobile_number: str, otp: str):
    otp_store[mobile_number] = {
        "otp": otp,
        "created_at": datetime.utcnow()
    }

def verify_otp(mobile_number: str, otp: str) -> bool:
    record = otp_store.get(mobile_number)
    if not record:
        return False
    if record["otp"] != otp:
        return False
    if datetime.utcnow() > record["created_at"] + timedelta(minutes=10):
        del otp_store[mobile_number]
        return False
    del otp_store[mobile_number]
    return True

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        mobile_number: str = payload.get("sub")
        if mobile_number is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.mobile_number == mobile_number).first()
    if user is None:
        raise credentials_exception

    return user