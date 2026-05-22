"""Fetch and store game-time weather for MLB outdoor stadiums via Open-Meteo.

Open-Meteo is free with no API key. Forecasts are accurate 7 days out
and update every hour. Historical data also available.

Wind direction relative to ballpark matters for run scoring:
- Wind blowing out toward OF → more HRs, higher run total
- Wind blowing in from CF → suppresses HRs, lower run total
We store raw bearing and compute the ballpark-relative factor in features.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import requests
from sqlalchemy.orm import Session

from src.core.logging import get_logger
from src.db.models import Game, Sport
from src.db.models.odds import GameWeather
from src.ingest.common import IngestResult

log = get_logger(__name__)

_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Bearing (degrees, 0=N clockwise) from home plate toward center field.
# Wind blowing in THIS direction = blowing out (helps HRs).
# Source: Statcast park geometry data / baseball-reference park factors.
_BALLPARK_CF_BEARING: dict[str, float] = {
    "NYY": 42,
    "BOS": 90,
    "CHC": 180,
    "WSH": 0,
    "BAL": 315,
    "TOR": 270,
    "NYM": 0,
    "PHI": 315,
    "ATL": 315,
    "MIA": 0,
    "CIN": 0,
    "PIT": 270,
    "STL": 0,
    "MIL": 270,
    "CHW": 315,
    "DET": 315,
    "CLE": 315,
    "MIN": 0,
    "KC": 270,
    "TEX": 45,
    "HOU": 315,
    "LAA": 315,
    "OAK": 270,
    "SEA": 315,
    "SF": 270,
    "LAD": 315,
    "ARI": 315,
    "COL": 315,
    "SD": 315,
    "TB": 0,
}

# Indoor/retractable-roof parks where weather is irrelevant.
_INDOOR_PARKS: set[str] = {"TB", "MIA", "HOU", "MIN", "ARI", "TOR", "SEA"}


def ingest_weather_for_upcoming(session: Session, lookahead_days: int = 5) -> IngestResult:
    """Fetch weather for all MLB games scheduled in the next N days."""
    from datetime import timedelta

    result = IngestResult()

    sport = session.query(Sport).filter_by(code="mlb").first()
    if sport is None:
        return result

    now = datetime.now(UTC)
    cutoff = now + timedelta(days=lookahead_days)

    games = (
        session.query(Game)
        .filter(
            Game.sport_id == sport.id,
            Game.scheduled_utc >= now,
            Game.scheduled_utc <= cutoff,
            Game.status != "final",
        )
        .all()
    )

    for game in games:
        # Skip if already have weather for this game.
        existing = session.query(GameWeather).filter_by(game_id=game.id).first()
        if existing and (now - existing.fetched_at).total_seconds() < 3600:
            continue

        venue = game.venue
        if venue is None or venue.lat is None or venue.lon is None:
            continue

        # Determine ballpark abbreviation from venue meta
        abbrev = (venue.meta or {}).get("team_abbrev", "")
        if abbrev in _INDOOR_PARKS:
            continue

        try:
            wx = _fetch_weather(venue.lat, venue.lon, game.scheduled_utc)
        except Exception as exc:
            log.warning("weather.fetch_failed", game_id=game.id, error=str(exc))
            result.errors.append(str(exc))
            continue

        if existing:
            existing.temp_f = wx.get("temp_f")
            existing.wind_mph = wx.get("wind_mph")
            existing.wind_bearing = wx.get("wind_bearing")
            existing.precip_prob = wx.get("precip_prob")
            existing.conditions = wx.get("conditions")
            existing.fetched_at = now
            result.rows_updated += 1
        else:
            session.add(
                GameWeather(
                    game_id=game.id,
                    temp_f=wx.get("temp_f"),
                    wind_mph=wx.get("wind_mph"),
                    wind_bearing=wx.get("wind_bearing"),
                    precip_prob=wx.get("precip_prob"),
                    conditions=wx.get("conditions"),
                    fetched_at=now,
                )
            )
            result.rows_inserted += 1

    session.flush()
    log.info("weather.ingest_done", inserted=result.rows_inserted, updated=result.rows_updated)
    return result


def _fetch_weather(lat: float, lon: float, game_utc: datetime) -> dict[str, Any]:
    date_str = game_utc.strftime("%Y-%m-%d")
    resp = requests.get(
        _OPEN_METEO_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,windspeed_10m,winddirection_10m,precipitation_probability,weathercode",
            "temperature_unit": "fahrenheit",
            "windspeed_unit": "mph",
            "timezone": "UTC",
            "start_date": date_str,
            "end_date": date_str,
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    hourly = data.get("hourly", {})

    hour = game_utc.hour
    temps = hourly.get("temperature_2m", [])
    winds = hourly.get("windspeed_10m", [])
    bearings = hourly.get("winddirection_10m", [])
    precips = hourly.get("precipitation_probability", [])
    codes = hourly.get("weathercode", [])

    idx = min(hour, len(temps) - 1) if temps else 0

    code = codes[idx] if idx < len(codes) else None
    return {
        "temp_f": temps[idx] if idx < len(temps) else None,
        "wind_mph": winds[idx] if idx < len(winds) else None,
        "wind_bearing": bearings[idx] if idx < len(bearings) else None,
        "precip_prob": (precips[idx] / 100.0) if idx < len(precips) else None,
        "conditions": _wmo_code_to_str(code),
    }


def _wmo_code_to_str(code: int | None) -> str:
    if code is None:
        return "unknown"
    if code == 0:
        return "clear"
    if code <= 3:
        return "cloudy"
    if code <= 67:
        return "rain"
    if code <= 77:
        return "snow"
    if code <= 99:
        return "storm"
    return "unknown"


def wind_out_factor(wind_mph: float, wind_bearing: float, park_abbrev: str) -> float:
    """Compute a wind factor: positive = helping HRs, negative = suppressing.

    Returns a value roughly in [-1, 1] where:
      +1 = strong wind directly out (toward CF)
      -1 = strong wind directly in (from CF)
       0 = crosswind or calm
    """
    import math

    cf_bearing = _BALLPARK_CF_BEARING.get(park_abbrev)
    if cf_bearing is None or wind_mph < 2:
        return 0.0

    # Angle between wind direction and CF bearing
    diff = abs(wind_bearing - cf_bearing) % 360
    if diff > 180:
        diff = 360 - diff

    # diff=0 means wind blowing toward CF (out), diff=180 means into CF (in)
    alignment = math.cos(math.radians(diff))  # +1=out, -1=in
    strength = min(wind_mph / 15.0, 1.0)  # cap at 15mph for normalisation
    return round(alignment * strength, 3)
