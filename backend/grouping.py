"""Split a trip into groups by time gaps and travel distance."""
from __future__ import annotations

import hashlib
import math
from datetime import datetime


def haversine_km(a, b) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def part_of_day(dt: datetime) -> str:
    h = dt.hour
    if h < 5:
        return "Night"
    if h < 11:
        return "Morning"
    if h < 16:
        return "Afternoon"
    if h < 19:
        return "Golden hour"
    return "Evening"


def group_id(clip_ids: list[str]) -> str:
    return hashlib.sha1("|".join(sorted(clip_ids)).encode()).hexdigest()[:10]


def summarize(clips: list[dict], first_day: datetime.date, title: str | None = None, kind: str = "moment", extra: dict | None = None) -> dict:
    clips = sorted(clips, key=lambda c: c["captured"])
    start = datetime.fromisoformat(clips[0]["captured"])
    end = datetime.fromisoformat(clips[-1]["captured"])
    gps = [c["gps"] for c in clips if c.get("gps")]
    center = [sum(g[0] for g in gps) / len(gps), sum(g[1] for g in gps) / len(gps)] if gps else None
    days = sorted({datetime.fromisoformat(c["captured"]).date() for c in clips})
    day_nums = [(d - first_day).days + 1 for d in days]
    if title is None:
        dn = f"Day {day_nums[0]}" if len(day_nums) == 1 else f"Days {day_nums[0]}–{day_nums[-1]}"
        title = f"{dn} · {start.strftime('%a %d %b')} · {part_of_day(start)}"
    g = {
        "id": group_id([c["id"] for c in clips]),
        "kind": kind,
        "title": title,
        "clip_ids": [c["id"] for c in clips],
        "start": start.isoformat(),
        "end": end.isoformat(),
        "days": day_nums,
        "center": center,
        "footage_seconds": round(sum(c["duration"] for c in clips), 1),
        "cameras": sorted({c["camera"] for c in clips}),
        "has_drone": any(c.get("is_drone") for c in clips),
        "portrait_share": round(sum(c["orientation"] == "portrait" for c in clips) / len(clips), 2),
    }
    if extra:
        g.update(extra)
    return g


def group_by_moment(clips: list[dict], gap_hours: float = 3.0, km: float = 10.0) -> list[dict]:
    if not clips:
        return []
    clips = sorted(clips, key=lambda c: c["captured"])
    first_day = datetime.fromisoformat(clips[0]["captured"]).date()
    runs: list[list[dict]] = [[clips[0]]]
    last_gps = clips[0].get("gps")
    for prev, cur in zip(clips, clips[1:]):
        t0 = datetime.fromisoformat(prev["captured"]) + _dur(prev)
        t1 = datetime.fromisoformat(cur["captured"])
        gap_h = (t1 - t0).total_seconds() / 3600
        moved = bool(cur.get("gps") and last_gps and haversine_km(cur["gps"], last_gps) > km)
        crossed_night = t1.date() != t0.date() and gap_h > 1.0
        if gap_h > gap_hours or moved or crossed_night:
            runs.append([cur])
        else:
            runs[-1].append(cur)
        if cur.get("gps"):
            last_gps = cur["gps"]
    return [summarize(r, first_day) for r in runs]


def _dur(c):
    from datetime import timedelta
    return timedelta(seconds=c["duration"])


def trip_first_day(clips: list[dict]):
    return min(datetime.fromisoformat(c["captured"]) for c in clips).date()
