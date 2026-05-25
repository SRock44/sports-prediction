"""Celery application instance and base task configuration."""

from __future__ import annotations

import contextlib
import threading
import time
from datetime import UTC, datetime

import requests
from celery import Celery
from celery.signals import task_failure, task_postrun, task_prerun, task_retry

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
        "src.tasks.outcome_tasks",
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
        "src.tasks.outcome_tasks.*": {"queue": "default"},
        # Tasks that need the high_priority queue pass queue= explicitly via apply_async.
    },
    beat_schedule_filename="celerybeat-schedule",
)

# Import beat schedule (sets app.conf.beat_schedule)
from src.tasks.schedule import BEAT_SCHEDULE  # noqa: E402

app.conf.beat_schedule = BEAT_SCHEDULE


# ── Task lifecycle webhook notifications ─────────────────────────────────────

# Track start times per task_id so we can report duration on completion.
_task_start: dict[str, float] = {}


def _short_name(task_name: str) -> str:
    """Strip module prefix: src.tasks.ingest_tasks.ingest_yesterday_nba → ingest_yesterday_nba"""
    return task_name.split(".")[-1]


def _ts() -> str:
    return datetime.now(UTC).strftime("%H:%M UTC")


def _post_ops(embed: dict) -> None:
    url = settings.discord_webhook_ops
    if not url:
        return
    with contextlib.suppress(Exception):
        requests.post(url, json={"embeds": [embed]}, timeout=5)


def _fire(embed: dict) -> None:
    threading.Thread(target=_post_ops, args=(embed,), daemon=True).start()


@task_prerun.connect
def on_task_start(task_id: str, task, args, kwargs, **_):
    _task_start[task_id] = time.monotonic()
    name = _short_name(task.name)
    kw_summary = ", ".join(f"{k}={v}" for k, v in (kwargs or {}).items())
    description = kw_summary or "no kwargs"
    _fire(
        {
            "title": f"▶️  STARTED  ·  {name}",
            "description": description,
            "color": 0x5865F2,  # blurple
            "footer": {"text": _ts()},
        }
    )


@task_postrun.connect
def on_task_done(task_id: str, task, args, kwargs, retval, state: str, **_):
    elapsed = time.monotonic() - _task_start.pop(task_id, time.monotonic())
    name = _short_name(task.name)
    if state == "SUCCESS":
        description = f"Finished in {elapsed:.1f}s"
        # For train_props: retval is {stat: {run_id, mae} | "error: ..."} — show per-stat MAE
        if name == "train_props" and isinstance(retval, dict):
            sport = (kwargs or {}).get("sport", "?")
            lines = [f"sport={sport}  elapsed={elapsed:.0f}s"]
            for stat, result in sorted(retval.items()):
                if isinstance(result, dict):
                    mae = result.get("mae", "?")
                    run = result.get("run_id", "?")
                    lines.append(f"  `{stat:<12}` MAE `{mae}`  run `{run}`")
                else:
                    lines.append(f"  `{stat:<12}` {result}")
            description = "\n".join(lines)
        _fire(
            {
                "title": f"✅  SUCCEEDED  ·  {name}",
                "description": description,
                "color": 0x57F287,  # green
                "footer": {"text": _ts()},
            }
        )
    elif state == "FAILURE":
        # task_failure signal fires too, but postrun catches non-exception failures
        pass


@task_failure.connect
def on_task_failure(task_id: str, exception, traceback, sender, **_):
    elapsed = time.monotonic() - _task_start.pop(task_id, time.monotonic())
    name = _short_name(sender.name)
    exc_str = f"{type(exception).__name__}: {exception}"
    _fire(
        {
            "title": f"❌  FAILED  ·  {name}",
            "description": f"```{exc_str[:300]}```\nElapsed: {elapsed:.1f}s",
            "color": 0xED4245,  # red
            "footer": {"text": _ts()},
        }
    )


@task_retry.connect
def on_task_retry(request, reason, einfo, **_):
    name = _short_name(request.task)
    _fire(
        {
            "title": f"🔁  RETRY  ·  {name}",
            "description": str(reason)[:200],
            "color": 0xFEE75C,  # yellow
            "footer": {"text": _ts()},
        }
    )
