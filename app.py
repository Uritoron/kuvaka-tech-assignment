from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from src.api.v1.auth import router as auth_router
from src.api.v1.user import router as user_router
from src.api.v1.chatroom import router as chatroom_router
from src.api.v1.subscription import router as subscription_router

from src.models.user import User
from src.models.chatroom import Chatroom, Message


from src.database.session import engine
from src.database.base import Base

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Gemini Backend Clone - Kuvaka Tech",
    version="1.0.0",
    description="OTP + JWT Auth, Async Gemini API, Stripe Subscriptions"
)

app.include_router(auth_router)
app.include_router(user_router)
app.include_router(chatroom_router)
app.include_router(subscription_router)

# Override OpenAPI schema to fix Swagger UI
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Replace default OAuth2 password flow with Bearer token
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Enter: **Bearer <your_jwt_token>**\n\nGet token via `POST /auth/verify-otp` after signing up and receiving OTP."
        }
    }

    # Apply BearerAuth to all protected endpoints
    for path in openapi_schema["paths"].values():
        for method in path.values():
            if "security" in method:
                method["security"] = [{"BearerAuth": []}]

    app.openapi_schema = openapi_schema
    return openapi_schema

app.openapi = custom_openapi