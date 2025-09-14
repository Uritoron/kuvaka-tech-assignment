# src/core/config.py
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()  # ✅ MUST BE FIRST

class Settings:
    PROJECT_NAME: str = "Gemini Backend Clone - Kuvaka Tech"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your_super_secret_key_here_123!")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 1440))

    DATABASE_URL: str = os.getenv("DATABASE_URL")

    # Redis
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    REDIS_DB: int = int(os.getenv("REDIS_DB", 0))
    REDIS_SSL: bool = os.getenv("REDIS_SSL", "false").lower() == "true"

    # Gemini
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Stripe
    STRIPE_SECRET_KEY: str = os.getenv("STRIPE_SECRET_KEY", "")
    STRIPE_WEBHOOK_SECRET: str = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")

settings = Settings()