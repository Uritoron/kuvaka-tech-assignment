# src/utils/cache.py
import redis
import json
from datetime import datetime 
from typing import Optional, List, Dict
from src.core.config import settings
import ssl

# ✅ Create a custom SSL context for compatibility
ssl_context = ssl.create_default_context()
# Force TLS 1.2 or higher (required by Redis Cloud)
ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
ssl_context.maximum_version = ssl.TLSVersion.TLSv1_3

# ✅ Initialize Redis client with explicit SSL context
redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    password=settings.REDIS_PASSWORD,
    db=settings.REDIS_DB
)

# --- CACHE UTILITIES ---
def cache_chatrooms(user_id: str, chatrooms: List[Dict]) -> None:
    """Cache user’s chatroom list with 5-minute TTL"""
    # ✅ Convert all datetime objects to ISO format strings
    def serialize_datetime(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    # Serialize each chatroom dict with custom serializer
    serialized_chatrooms = [
        json.loads(json.dumps(chatroom, default=serialize_datetime))
        for chatroom in chatrooms
    ]

    key = f"chatrooms:user:{user_id}"
    redis_client.setex(key, 300, json.dumps(serialized_chatrooms))

def get_cached_chatrooms(user_id: str) -> Optional[List[Dict]]:
    """Get cached chatrooms, return None if not found"""
    key = f"chatrooms:user:{user_id}"
    data = redis_client.get(key)
    if data:
        return json.loads(data)
    return None

def invalidate_chatrooms_cache(user_id: str) -> None:
    """Delete cache when chatroom is created/deleted"""
    key = f"chatrooms:user:{user_id}"
    redis_client.delete(key)