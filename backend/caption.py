"""Post caption + hashtags for Instagram / TikTok / Shorts: Claude when available, an offline template otherwise."""
from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, Field

from . import ai
from .director import PLATFORMS
from .looks import LOOKS

PLATFORM_RULES = {
    "reels": "Instagram Reels: 1-3 short lines, a hook first, emoji welcome, 5-8 hashtags.",
    "feed": "Instagram feed post: 1-3 short lines, emoji welcome, 5-8 hashtags.",
    "square": "Instagram post: 1-3 short lines, emoji welcome, 5-8 hashtags.",
    "tiktok": "TikTok: one punchy line (under 100 characters), 3-5 hashtags.",
    "shorts": "YouTube Shorts: a title-like line under 80 characters, 3 hashtags including #shorts.",
    "youtube": "YouTube video: a descriptive title line and a two-sentence description, 3-5 hashtags.",
    "youtube4k": "YouTube video: a descriptive title line and a two-sentence description, 3-5 hashtags.",
    "vertical": "Instagram Reels / TikTok: 1-2 short lines, a hook first, 5 hashtags.",
}


class PostCaption(BaseModel):
    caption: str = Field(description="The post text, without hashtags")
    hashtags: list[str] = Field(description="Hashtags without the # sign")


def _dates(plan: dict) -> str:
    days = sorted({s["captured"][:10] for s in plan["shots"]})
    if not days:
        return ""
    a, b = datetime.fromisoformat(days[0]), datetime.fromisoformat(days[-1])
    if a.date() == b.date():
        return a.strftime("%-d %B %Y")
    if a.month == b.month:
        return f"{a.day}–{b.strftime('%-d %B %Y')}"
    return f"{a.strftime('%-d %b')} – {b.strftime('%-d %b %Y')}"


def _tag(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z]", "", s.title().replace(" ", ""))


def template(plan: dict, group: dict, clips: dict) -> dict:
    text = plan.get("text") or {}
    place = (text.get("place") or "").strip()
    notes = [v.strip() for _, v in sorted((text.get("notes") or {}).items(), key=lambda kv: int(kv[0])) if v and v.strip()]
    platform = plan.get("platform", "reels")
    head = " · ".join(x for x in (place, _dates(plan)) if x) or group.get("title", "")
    lines = [head]
    if notes:
        lines.append(" · ".join(notes[:3]))
    elif group.get("description"):
        lines.append(group["description"])
    tags = []
    if place:
        tags += [_tag(place), f"{_tag(place)}Travel"]
    scenes = {clips[s["clip_id"]].get("tags", {}).get("scene", "") for s in plan["shots"] if s["clip_id"] in clips}
    tags += [_tag(x) for x in sorted(scenes) if x][:3]
    tags += {"tiktok": ["TikTokTravel", "TravelTok"], "shorts": ["Shorts", "TravelShorts"]}.get(platform, ["TravelReels", "Reels"])
    if plan["look"] in ("cinematic", "film", "noir"):
        tags.append("Cinematic")
    tags += ["Travel", "TravelVlog"]
    seen, out = set(), []
    for t in tags:
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    limit = {"tiktok": 5, "shorts": 3}.get(platform, 8)
    return {"caption": "\n".join(lines), "hashtags": out[:limit], "source": "template"}


def with_claude(plan: dict, group: dict, clips: dict) -> dict:
    text = plan.get("text") or {}
    shots = []
    for s in plan["shots"]:
        t = clips.get(s["clip_id"], {}).get("tags") or {}
        if t.get("caption"):
            shots.append(f"- {t.get('scene', '')}: {t['caption']}")
    notes = [v for _, v in sorted((text.get("notes") or {}).items(), key=lambda kv: int(kv[0])) if v and v.strip()]
    platform = plan.get("platform", "reels")
    facts = [
        f"Platform: {PLATFORMS.get(platform, {}).get('name', platform)}",
        f"Rules: {PLATFORM_RULES.get(platform, PLATFORM_RULES['reels'])}",
        f"Place: {text.get('place') or 'not given (do not guess a place name)'}",
        f"Dates: {_dates(plan)}",
        f"Group: {group.get('title', '')} {('- ' + group['description']) if group.get('description') else ''}",
        f"Look: {LOOKS[plan['look']].name} ({LOOKS[plan['look']].blurb})",
        f"Length: {round(plan['duration'])} seconds",
    ]
    if notes:
        facts.append("On-screen notes: " + " | ".join(notes))
    if shots:
        facts.append("What the shots show:\n" + "\n".join(dict.fromkeys(shots)))
    prompt = ("Write the post caption for this short travel video. Sound like a real person sharing their trip, "
              "not an ad. Only use facts given below; don't invent places, food or events.\n\n" + "\n".join(facts))
    res = ai._parse(ai._client(), [{"type": "text", "text": prompt}], PostCaption, max_tokens=2000)
    tags = [re.sub(r"[^0-9A-Za-z_]", "", h.lstrip("#")) for h in res.hashtags]
    return {"caption": res.caption.strip(), "hashtags": [t for t in tags if t], "source": "claude"}


def write(plan: dict, group: dict, clips: dict, use_ai: bool = True) -> dict:
    if use_ai and ai.available():
        try:
            return with_claude(plan, group, clips)
        except Exception as e:  # fall back so the button always produces something
            out = template(plan, group, clips)
            out["note"] = f"Claude wasn't available ({str(e)[:120]}), so this is a template caption."
            return out
    out = template(plan, group, clips)
    if use_ai:
        out["note"] = "No Claude API key set, so this is a template caption."
    return out
