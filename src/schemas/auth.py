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


class ForgotPasswordRequest(BaseModel):
    mobile_number: str

# --- UPDATED: ChangePasswordRequest ---
class ChangePasswordRequest(BaseModel):
    # The user only needs to provide the NEW password.
    # The old password is not required because the user is already authenticated via JWT.
    new_password: str = Field(..., min_length=8)  # Enforce a minimum password length for security
