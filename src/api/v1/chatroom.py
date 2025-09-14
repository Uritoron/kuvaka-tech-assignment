# src/api/v1/chatroom.py
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
from google.generativeai import generative_models as genai
from src.schemas.chatroom import ChatroomCreate, ChatroomResponse, MessageCreate, MessageResponse
from src.models import Chatroom, Message, User
from src.database.session import get_db
from src.core.config import settings
from src.core.security import get_current_user
from src.utils.cache import get_cached_chatrooms, cache_chatrooms, invalidate_chatrooms_cache
import logging

router = APIRouter(prefix="/chatroom", tags=["chatroom"])
logger = logging.getLogger(__name__)


async def process_gemini_message(
    message_content: str,
    chatroom_id: int, 
    user_id: int,
    db: Session
):
    """Process Gemini API call in background"""
    try:
        # ✅ Configure Gemini API key INSIDE the background task
        genai.configure(api_key=settings.GEMINI_API_KEY)
        
        # Validate ownership
        user = db.query(User).filter(User.id == user_id).first()
        chatroom = db.query(Chatroom).filter(Chatroom.id == chatroom_id).first()
        
        if not user or not chatroom:
            logger.error(f"❌ User or chatroom not found: user={user_id}, chatroom={chatroom_id}")
            return
        
        # Call Gemini API
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(message_content)
        
        # Save AI response to database
        ai_message = Message(
            content=response.text,
            is_from_user=False,
            chatroom_id=chatroom_id,
            user_id=user_id
        )
        db.add(ai_message)
        db.commit()
        db.refresh(ai_message)
        
        logger.info(f"✅ AI response saved for chatroom {chatroom_id}: {response.text[:50]}...")
        
    except Exception as e:
        logger.error(f"❌ Error in background task: {e}")
        # Don't raise - this would crash the background task
    finally:
        db.close()


# --- GET /chatroom — CACHED LIST OF CHATROOMS ---
@router.get("", response_model=List[ChatroomResponse])
def list_chatrooms(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Lists all chatrooms for the authenticated user.
    
    📌 CACHING JUSTIFICATION:
    - This endpoint is frequently accessed when loading the dashboard.
    - Chatrooms change infrequently compared to messages.
    - Caching the list with a 5-minute TTL significantly reduces database load.
    - Uses Redis for low-latency, persistent storage.
    """
    # Try cache first
    cached = get_cached_chatrooms(str(current_user.id))
    if cached:
        logger.info(f"✅ Serving cached chatrooms for user {current_user.id}")
        return [ChatroomResponse(**c) for c in cached]

    # Fallback to DB
    chatrooms = db.query(Chatroom).filter(Chatroom.user_id == current_user.id).all()

    # Cache result
    chatroom_schemas = [ChatroomResponse.from_orm(c) for c in chatrooms]
    cache_chatrooms(str(current_user.id), [c.model_dump() for c in chatroom_schemas])

    logger.info(f"✅ Cached new chatroom list for user {current_user.id}")
    return chatroom_schemas


# --- POST /chatroom — CREATE NEW CHATROOM ---
@router.post("", response_model=ChatroomResponse, status_code=status.HTTP_201_CREATED)
def create_chatroom(
    chatroom: ChatroomCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Creates a new chatroom for the authenticated user.
    Invalidates the chatroom cache on creation.
    """
    new_chat = Chatroom(name=chatroom.name, user_id=current_user.id)
    db.add(new_chat)
    db.commit()
    db.refresh(new_chat)

    # Invalidate cache to ensure fresh data on next GET /chatroom
    invalidate_chatrooms_cache(str(current_user.id))
    logger.info(f"✅ Created chatroom {new_chat.id} and invalidated cache for user {current_user.id}")

    return new_chat


# --- GET /chatroom/{chatroom_id} — GET SINGLE CHATROOM DETAILS ---
@router.get("/{chatroom_id}", response_model=ChatroomResponse)
def get_chatroom(
    chatroom_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves detailed information about a specific chatroom.
    Ensures ownership before returning.
    """
    chatroom = db.query(Chatroom).filter(
        Chatroom.id == chatroom_id,
        Chatroom.user_id == current_user.id
    ).first()

    if not chatroom:
        raise HTTPException(status_code=404, detail="Chatroom not found")

    return chatroom


# --- DELETE /chatroom/{chatroom_id} — DELETE CHATROOM ---
@router.delete("/{chatroom_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chatroom(
    chatroom_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Deletes a specific chatroom owned by the user.
    Invalidates the chatroom cache on deletion.
    """
    chatroom = db.query(Chatroom).filter(
        Chatroom.id == chatroom_id,
        Chatroom.user_id == current_user.id
    ).first()

    if not chatroom:
        raise HTTPException(status_code=404, detail="Chatroom not found")

    db.delete(chatroom)
    db.commit()

    # Invalidate cache to reflect deletion
    invalidate_chatrooms_cache(str(current_user.id))
    logger.info(f"✅ Deleted chatroom {chatroom_id} and invalidated cache for user {current_user.id}")


# --- POST /chatroom/{chatroom_id}/message — SEND MESSAGE TO GEMINI (ASYNC) ---
@router.post("/{chatroom_id}/message", response_model=MessageResponse, status_code=status.HTTP_202_ACCEPTED)
def send_message(
    chatroom_id: int,
    message_data: MessageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks()  # <-- Inject dependency
):
    """
    Sends a user message and triggers an async Gemini API call via FastAPI BackgroundTasks.
    Returns 202 Accepted immediately while processing occurs in background.
    
    🔒 RATE LIMITING:
    - Basic tier: 5 messages/day
    - Pro tier: Unlimited
    - Daily counter resets at UTC midnight
    """
    # Validate chatroom ownership
    chatroom = db.query(Chatroom).filter(
        Chatroom.id == chatroom_id,
        Chatroom.user_id == current_user.id
    ).first()

    if not chatroom:
        raise HTTPException(status_code=404, detail="Chatroom not found")

    # 🔒 RATE LIMITING: Basic tier = 5 messages/day
    now = datetime.utcnow().date()
    if current_user.subscription_tier == "Basic":
        # Reset counter if date changed
        if current_user.last_message_date.date() < now:
            current_user.daily_message_count = 0
            current_user.last_message_date = now
            db.commit()

        # Enforce limit
        if current_user.daily_message_count >= 5:
            raise HTTPException(
                status_code=429,
                detail="Daily message limit reached (5/day for Basic tier). Upgrade to Pro."
            )

        # Increment count
        current_user.daily_message_count += 1
        db.commit()

    # Save user message to DB
    user_message = Message(
        content=message_data.content,
        is_from_user=True,
        chatroom_id=chatroom_id,
        user_id=current_user.id
    )
    db.add(user_message)
    db.commit()
    db.refresh(user_message)

    # ✅ TRIGGER ASYNC GEMINI CALL USING BACKGROUND TASKS (NO CELERY!)
    background_tasks.add_task(
        process_gemini_message,
        message_content=message_data.content,
        chatroom_id=chatroom_id,
        user_id=current_user.id,
        db=db
    )

    logger.info(f"✅ Async Gemini task triggered for message {user_message.id} from user {current_user.id}")

    # Return 202 Accepted immediately — async processing in progress
    return user_message


# --- GET /chatroom/{chatroom_id}/messages — FETCH ALL MESSAGES IN CHATROOM ---
@router.get("/{chatroom_id}/messages", response_model=List[MessageResponse])
def get_messages(
    chatroom_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves all messages in a specific chatroom, ordered chronologically.
    Includes both user messages and AI responses.
    """
    chatroom = db.query(Chatroom).filter(
        Chatroom.id == chatroom_id,
        Chatroom.user_id == current_user.id
    ).first()

    if not chatroom:
        raise HTTPException(status_code=404, detail="Chatroom not found")

    messages = db.query(Message).filter(
        Message.chatroom_id == chatroom_id
    ).order_by(Message.created_at).all()

    return messages