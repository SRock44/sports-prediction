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


# Coordinates for all 30 MLB outdoor + retractable-roof parks.
_BALLPARK_COORDS: dict[str, tuple[float, float]] = {
    "ARI": (33.4455, -112.0667),
    "ATL": (33.8908, -84.4678),
    "BAL": (39.2839, -76.6216),
    "BOS": (42.3467, -71.0972),
    "CHC": (41.9484, -87.6553),
    "CWS": (41.8299, -87.6338),
    "CIN": (39.0979, -84.5082),
    "CLE": (41.4962, -81.6852),
    "COL": (39.7559, -104.9942),
    "DET": (42.3390, -83.0485),
    "HOU": (29.7573, -95.3555),
    "KC": (39.0517, -94.4803),
    "LAA": (33.8003, -117.8827),
    "LAD": (34.0739, -118.2400),
    "MIA": (25.7781, -80.2197),
    "MIL": (43.0280, -87.9712),
    "MIN": (44.9817, -93.2776),
    "NYM": (40.7571, -73.8458),
    "NYY": (40.8296, -73.9262),
    "OAK": (37.7516, -122.2005),
    "PHI": (39.9056, -75.1665),
    "PIT": (40.4469, -80.0057),
    "SD": (32.7076, -117.1570),
    "SF": (37.7786, -122.3893),
    "SEA": (47.5914, -122.3325),
    "STL": (38.6226, -90.1928),
    "TB": (27.7683, -82.6534),
    "TEX": (32.7513, -97.0831),
    "TOR": (43.6414, -79.3894),
    "WSH": (38.8730, -77.0074),
}

_OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def populate_venue_coords(session: Session) -> int:
    """One-time: populate lat/lon on all MLB venue records from the known coords dict."""
    from src.db.models import Sport, Venue

    sport = session.query(Sport).filter_by(code="mlb").first()
    if sport is None:
        return 0

    updated = 0
    venues = session.query(Venue).filter_by(sport_id=sport.id).all()
    for v in venues:
        abbrev = (v.meta or {}).get("team_abbrev", "")
        coords = _BALLPARK_COORDS.get(abbrev)
        if coords and (v.lat is None or v.lon is None):
            v.lat, v.lon = coords
            updated += 1
    session.flush()
    return updated


def ingest_weather_historical(session: Session, season_from: int = 2022) -> IngestResult:
    """Backfill game-time weather for all historical final MLB games using Open-Meteo archive.

    Groups games by (venue, date) to minimise API calls — one request per venue-day.
    Skips indoor parks and games that already have weather stored.
    """
    import time
    from collections import defaultdict

    result = IngestResult()
    sport = session.query(Sport).filter_by(code="mlb").first()
    if sport is None:
        return result

    games = (
        session.query(Game)
        .join(Game.venue)
        .filter(
            Game.sport_id == sport.id,
            Game.status == "final",
            Game.season >= season_from,
            Game.venue_id.isnot(None),
        )
        .all()
    )

    # Skip games that already have weather stored
    from src.db.models.odds import GameWeather

    existing_ids = {gw.game_id for gw in session.query(GameWeather.game_id).all()}

    # Group by (venue_abbrev, date) to batch API calls
    venue_date_games: dict[tuple[str, str, float, float], list[Game]] = defaultdict(list)
    for game in games:
        if game.id in existing_ids:
            continue
        venue = game.venue
        if venue is None or venue.lat is None or venue.lon is None:
            continue
        abbrev = (venue.meta or {}).get("team_abbrev", "")
        if abbrev in _INDOOR_PARKS:
            continue
        date_str = game.scheduled_utc.strftime("%Y-%m-%d")
        venue_date_games[(abbrev, date_str, venue.lat, venue.lon)].append(game)

    log.info("weather_backfill.start", groups=len(venue_date_games))

    for (abbrev, date_str, lat, lon), day_games in venue_date_games.items():
        try:
            resp = requests.get(
                _OPEN_METEO_ARCHIVE_URL,
                params={
                    "latitude": str(lat),
                    "longitude": str(lon),
                    "hourly": "temperature_2m,windspeed_10m,winddirection_10m,precipitation_probability,weathercode",
                    "temperature_unit": "fahrenheit",
                    "windspeed_unit": "mph",
                    "timezone": "UTC",
                    "start_date": date_str,
                    "end_date": date_str,
                },
                timeout=15,
            )
            resp.raise_for_status()
            hourly = resp.json().get("hourly", {})

            for game in day_games:
                hour = min(game.scheduled_utc.hour, 23)
                temps = hourly.get("temperature_2m", [])
                winds = hourly.get("windspeed_10m", [])
                bearings = hourly.get("winddirection_10m", [])
                precips = hourly.get("precipitation_probability", [])
                codes = hourly.get("weathercode", [])

                idx = min(hour, len(temps) - 1) if temps else 0
                code = codes[idx] if idx < len(codes) else None

                raw_precip = precips[idx] if idx < len(precips) else None
                session.add(
                    GameWeather(
                        game_id=game.id,
                        temp_f=temps[idx] if idx < len(temps) else None,
                        wind_mph=winds[idx] if idx < len(winds) else None,
                        wind_bearing=bearings[idx] if idx < len(bearings) else None,
                        precip_prob=(raw_precip / 100.0) if raw_precip is not None else None,
                        conditions=_wmo_code_to_str(code),
                        fetched_at=datetime.now(UTC),
                    )
                )
                result.rows_inserted += 1

            time.sleep(0.05)  # ~20 req/s — well within Open-Meteo free limits
        except Exception as exc:
            log.warning("weather_backfill.error", abbrev=abbrev, date=date_str, error=str(exc))
            result.errors.append(str(exc))

    session.flush()
    log.info("weather_backfill.done", inserted=result.rows_inserted, errors=len(result.errors))
    return result


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
        params={  # type: ignore[arg-type]
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
