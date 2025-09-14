# celery_worker.py (in project root)
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tasks import celery_app

if __name__ == '__main__':
    celery_app.start()
