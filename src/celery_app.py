# src/celery_app.py
from celery import Celery
import google.generativeai as genai
from src.core.config import settings
from src.database.session import SessionLocal
from src.models import Message, User, Chatroom

# Configure Celery with Redis
celery_app = Celery(
    'gemini_tasks',
    broker=f'redis://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}',
    backend=f'redis://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}'
)

# Configure Gemini API
genai.configure(api_key=settings.GEMINI_API_KEY)

@celery_app.task
def process_gemini_message(message_content: str, chatroom_id: int, user_id: int):
    """Process Gemini API call as a Celery task"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        chatroom = db.query(Chatroom).filter(Chatroom.id == chatroom_id).first()

        if not user or not chatroom:
            print(f"❌ User or chatroom not found: user={user_id}, chatroom={chatroom_id}")
            return

        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(message_content)

        ai_message = Message(
            content=response.text,
            is_from_user=False,
            chatroom_id=chatroom_id,
            user_id=user_id
        )
        db.add(ai_message)
        db.commit()
        db.refresh(ai_message)

        print(f"✅ AI response saved for chatroom {chatroom_id}: {response.text[:50]}...")
        return {"success": True, "message_id": ai_message.id}

    except Exception as e:
        print(f"❌ Error in Celery task: {e}")
        raise
    finally:
        db.close()
