"""Optional Claude vision: tag each clip's scene, then propose theme groups across the whole trip."""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Callable, Literal

import anthropic
from pydantic import BaseModel, Field

from .config import AI_MODEL

BATCH = 8


class ClipTag(BaseModel):
    clip_id: str
    scene: str = Field(description="Short scene label, e.g. 'beach', 'street food', 'temple', 'rice terraces', 'city at night'")
    setting: Literal["outdoor", "indoor", "aerial", "underwater", "vehicle"]
    caption: str = Field(description="What happens in the clip, max 12 words")
    people: Literal["none", "few", "crowd", "selfie"]
    highlight: int = Field(description="0-10: how visually striking this clip is for a travel reel")


class TagBatch(BaseModel):
    clips: list[ClipTag]


class ThemeGroup(BaseModel):
    title: str = Field(description="Short evocative title, max 5 words")
    description: str = Field(description="One sentence on what ties these clips together")
    clip_ids: list[str]
    suggested_look: Literal["cinematic", "vibrant", "moody", "film", "golden", "clean", "noir"]
    music_mood: str = Field(description="2-4 words, e.g. 'chill tropical house'")


class ThemePlan(BaseModel):
    groups: list[ThemeGroup]


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def _parse(client, content, schema, max_tokens=8000):
    try:
        return _call(client, content, schema, max_tokens)
    except anthropic.AuthenticationError:
        raise RuntimeError("Claude rejected the API key. Start VideoCut with a valid ANTHROPIC_API_KEY to use theme grouping.")
    except anthropic.RateLimitError:
        raise RuntimeError("Claude is rate-limiting right now. Try again in a minute.")
    except anthropic.APIConnectionError:
        raise RuntimeError("Couldn't reach Claude. Check your internet connection.")


def _call(client, content, schema, max_tokens):
    resp = client.beta.messages.parse(
        model=AI_MODEL,
        max_tokens=max_tokens,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "low"},
        output_format=schema,
        messages=[{"role": "user", "content": content}],
    )
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        raise RuntimeError(f"Claude didn't return tags (stop_reason={resp.stop_reason})")
    return resp.parsed_output


def tag_clips(clips: list[dict], sheet_for: Callable[[dict], Path], progress: Callable[[float, str], None] | None = None) -> dict[str, dict]:
    client = _client()
    out: dict[str, dict] = {}
    batches = [clips[i:i + BATCH] for i in range(0, len(clips), BATCH)]
    for bi, batch in enumerate(batches):
        content: list[dict] = []
        for c in batch:
            sheet = sheet_for(c)
            if not sheet.exists():
                continue
            content.append({"type": "text", "text": f"clip_id={c['id']} ({c['camera']}, {c['duration']:.0f}s, captured {c['captured'][:16]})"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                         "data": base64.standard_b64encode(sheet.read_bytes()).decode()}})
        if not content:
            continue
        content.append({"type": "text", "text": (
            "Each image is a 2x2 contact sheet of four frames from one travel video clip, labelled by the clip_id above it. "
            "Tag every clip so a video editor can group them by theme and pick highlights. Return one entry per clip_id.")})
        result = _parse(client, content, TagBatch)
        for tag in result.clips:
            out[tag.clip_id] = tag.model_dump()
        if progress:
            progress((bi + 1) / len(batches), f"Tagged {min(len(clips), (bi + 1) * BATCH)}/{len(clips)} clips")
    return out


def theme_groups(clips: list[dict], tags: dict[str, dict]) -> list[dict]:
    lines = []
    for c in sorted(clips, key=lambda c: c["captured"]):
        t = tags.get(c["id"], {})
        gps = f"{c['gps'][0]:.3f},{c['gps'][1]:.3f}" if c.get("gps") else "no-gps"
        lines.append(f"{c['id']} | {c['captured'][:16]} | {gps} | {c['camera']} | {t.get('scene', '?')} | "
                     f"{t.get('setting', '?')} | {t.get('caption', '')} | highlight {t.get('highlight', '?')}")
    prompt = (
        "You are planning short social videos from a travel trip. Below is every clip: id | time | gps | camera | scene | "
        "setting | caption | highlight score.\n\n" + "\n".join(lines) + "\n\n"
        "Group the clips into themed stories that would each make a good 10-30 second reel. A theme can span different "
        "days and places (e.g. all the beaches, all the food, temples, a road trip, nightlife, drone views of the coast). "
        "Prefer 2-8 groups with at least 3 clips each where possible. Every clip_id must appear in exactly one group. "
        "Pick the colour look that suits each group and a music mood."
    )
    plan = _parse(_client(), [{"type": "text", "text": prompt}], ThemePlan)
    known = {c["id"] for c in clips}
    seen: set[str] = set()
    groups = []
    for g in plan.groups:
        ids = [i for i in g.clip_ids if i in known and i not in seen]
        seen.update(ids)
        if ids:
            groups.append({**g.model_dump(), "clip_ids": ids})
    leftover = [c["id"] for c in clips if c["id"] not in seen]
    if leftover:
        groups.append({"title": "More moments", "description": "Clips that didn't fit a theme", "clip_ids": leftover,
                       "suggested_look": "clean", "music_mood": ""})
    return groups
