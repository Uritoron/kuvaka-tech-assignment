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

# Initialize Passlib context for hashing (supports bcrypt)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# In-memory OTP store — Use Redis in production
otp_store: Dict[str, Dict[str, Any]] = {}

# Define OAuth2 scheme — matches our verify-otp endpoint
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/verify-otp")

# --- OTP FUNCTIONS ---
def generate_otp() -> str:
    """Generates a 6-digit numeric OTP."""
    return f"{secrets.randbelow(1000000):06}"

def store_otp(mobile_number: str, otp: str):
    """Stores an OTP with its creation timestamp."""
    otp_store[mobile_number] = {
        "otp": otp,
        "created_at": datetime.utcnow()
    }

def verify_otp(mobile_number: str, otp: str) -> bool:
    """Verifies an OTP for a given mobile number. Returns True if valid, False otherwise."""
    record = otp_store.get(mobile_number)
    if not record:
        return False
    if record["otp"] != otp:
        return False
    # Check if OTP is expired (e.g., 10 minutes)
    if datetime.utcnow() > record["created_at"] + timedelta(minutes=10):
        # Expired OTP, remove it from store
        del otp_store[mobile_number]
        return False
    # Valid OTP, remove it (OTP is single-use)
    del otp_store[mobile_number]
    return True

# --- PASSWORD HASHING FUNCTIONS ---
def get_password_hash(password: str) -> str:
    """
    Hashes a plaintext password using the configured scheme (bcrypt).
    
    Args:
        password (str): The plaintext password to hash.
        
    Returns:
        str: The hashed password.
    """
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies a plaintext password against a hashed password.
    
    Args:
        plain_password (str): The plaintext password provided by the user.
        hashed_password (str): The hashed password stored in the database.
        
    Returns:
        bool: True if the password matches the hash, False otherwise.
    """
    return pwd_context.verify(plain_password, hashed_password)

# --- JWT FUNCTIONS ---
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Creates a JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    """
    Dependency to get the current authenticated user from the JWT token.
    Raises an HTTPException if the token is invalid or the user doesn't exist.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # Decode the JWT token
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        # Extract the mobile_number (which we stored as 'sub')
        mobile_number: str = payload.get("sub")
        if mobile_number is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # Fetch the user from the database
    user = db.query(User).filter(User.mobile_number == mobile_number).first()
    if user is None:
        raise credentials_exception

    return user
