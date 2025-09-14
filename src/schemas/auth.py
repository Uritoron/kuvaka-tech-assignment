# src/schemas/auth.py
from pydantic import BaseModel, Field
from typing import Optional

class UserCreate(BaseModel):
    mobile_number: str = Field(..., min_length=10, max_length=15)

class OTPRequest(BaseModel):
    mobile_number: str = Field(..., min_length=10, max_length=15)

class OTPVerify(BaseModel):
    mobile_number: str = Field(..., min_length=10, max_length=15)
    otp: str = Field(..., min_length=4, max_length=6)

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    mobile_number: Optional[str] = None