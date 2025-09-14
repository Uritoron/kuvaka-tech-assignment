# src/api/v1/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from src.schemas.auth import UserCreate, OTPRequest, OTPVerify, Token
from src.models.user import User
from src.database.session import get_db
from src.core.security import (
    generate_otp,
    store_otp,
    verify_otp,
    create_access_token,
)

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/signup", response_model=UserCreate, status_code=status.HTTP_201_CREATED)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.mobile_number == user.mobile_number).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Mobile number already registered")
    new_user = User(mobile_number=user.mobile_number)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@router.post("/send-otp")
def send_otp(request: OTPRequest):
    otp = generate_otp()
    store_otp(request.mobile_number, otp)
    return {
        "message": "OTP generated successfully. (In real app, sent via SMS)",
        "otp": otp
    }

@router.post("/verify-otp", response_model=Token)
def verify_otp_endpoint(otp_data: OTPVerify, db: Session = Depends(get_db)):
    if not verify_otp(otp_data.mobile_number, otp_data.otp):
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")

    user = db.query(User).filter(User.mobile_number == otp_data.mobile_number).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found. Please signup first.")

    access_token = create_access_token(data={"sub": user.mobile_number})
    return {"access_token": access_token, "token_type": "bearer"}