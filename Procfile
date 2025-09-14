web: uvicorn app:app --host 0.0.0.0 --port $PORT
worker: cd src && celery -A src.celery_app.celery_app worker --loglevel=info --pool=solo