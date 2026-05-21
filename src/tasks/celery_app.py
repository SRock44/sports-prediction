"""Celery application instance and base task configuration."""
from __future__ import annotations

from celery import Celery
from celery.utils.log import get_task_logger

from src.core.config import settings
from src.core.logging import configure_logging

configure_logging()

app = Celery(
    "prediction",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "src.tasks.ingest_tasks",
        "src.tasks.feature_tasks",
        "src.tasks.train_tasks",
        "src.tasks.score_tasks",
    ],
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_routes={
        "src.tasks.train_tasks.*": {"queue": "default"},
        "src.tasks.ingest_tasks.*": {"queue": "default"},
        "src.tasks.feature_tasks.*": {"queue": "default"},
        "src.tasks.score_tasks.*": {"queue": "default"},
        # Tasks that need the high_priority queue pass queue= explicitly via apply_async.
    },
    beat_schedule_filename="celerybeat-schedule",
)

# Import beat schedule (sets app.conf.beat_schedule)
from src.tasks.schedule import BEAT_SCHEDULE  # noqa: E402
app.conf.beat_schedule = BEAT_SCHEDULE
