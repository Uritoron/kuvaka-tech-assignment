# src/api/v1/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from src.schemas.auth import UserCreate, OTPRequest, OTPVerify, Token, ForgotPasswordRequest, ChangePasswordRequest
from src.models.user import User
from src.database.session import get_db
from src.core.security import (
    generate_otp,
    store_otp,
    verify_otp,
    create_access_token,
    get_password_hash,
    verify_password,
    get_current_user
)

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/signup", response_model=UserCreate, status_code=status.HTTP_201_CREATED)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.mobile_number == user.mobile_number).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Mobile number already registered")
    
    # Prepare user data for creation
    user_data = {
        "mobile_number": user.mobile_number
    }
    # If a password is provided during signup, hash and store it
    if user.password:
        user_data["hashed_password"] = get_password_hash(user.password)
    # If no password is provided, hashed_password will be NULL/default in DB
    
    new_user = User(**user_data) # Create user instance with prepared data
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@router.post("/send-otp")
def send_otp(request: OTPRequest):
    # Note: In a real app, you might want to check if the user exists for 'forgot-password' flow
    # For simplicity here, we just generate/store it regardless.
    otp = generate_otp()
    store_otp(request.mobile_number, otp)
    return {
        "message": "OTP generated successfully. (In real app, sent via SMS)",
        "otp": otp # For assignment/demo purposes only
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

# --- POST /auth/forgot-password ---
@router.post("/forgot-password")
def forgot_password(request: ForgotPasswordRequest):
    """
    Sends an OTP to the user's mobile number for password reset.
    Note: This assumes the mobile number exists in the database.
    """
    # In a real app, you'd likely verify the mobile number exists first.
    # For this assignment, we'll just generate and store the OTP.
    # You could add a check here if desired.
    otp = generate_otp()
    store_otp(request.mobile_number, otp) # Store using the same mechanism
    return {
        "message": "OTP sent for password reset. (In real app, sent via SMS)",
        "otp": otp # For assignment/demo purposes only
    }

@router.post("/change-password", status_code=status.HTTP_200_OK)
def change_password(
    request: ChangePasswordRequest, # <-- Now only takes 'new_password'
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user) # <-- Already authenticated via JWT
):
    """
    Allows the authenticated user to change their password.
    Requires no old password as the user is already verified via JWT.
    """
    # Hash the new password
    new_hashed_password = get_password_hash(request.new_password)

    # Update user's password in the database
    current_user.hashed_password = new_hashed_password
    db.commit()

    return {"message": "Password changed successfully."}

# Import get_current_user at the bottom to avoid circular import issues
# if get_current_user depends on this router/file.
from src.core.security import get_current_user # Make sure this import is correct in your project structure