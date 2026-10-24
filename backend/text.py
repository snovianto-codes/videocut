"""Text on video: per-shot notes and a place / date stamp, styled per look and kept out of app UI zones."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

from .director import PLATFORMS

FONTS = "/System/Library/Fonts"
SUPP = f"{FONTS}/Supplemental"
EMOJI = f"{FONTS}/Apple Color Emoji.ttc"

# (note font, note index, stamp font, stamp index, uppercase notes?, pill?)
STYLES = {
    "minimal": (f"{FONTS}/Avenir Next.ttc", 2, f"{FONTS}/Avenir Next.ttc", 0, False, False),
    "pill": (f"{SUPP}/Futura.ttc", 2, f"{SUPP}/Futura.ttc", 4, False, True),
    "serif": (f"{SUPP}/Baskerville.ttc", 2, f"{SUPP}/Didot.ttc", 0, False, False),
}
LOOK_STYLE = {"vibrant": "pill", "film": "serif", "golden": "serif"}

# Flags and joined emoji need libraqm (Pillow finds it when Homebrew's fribidi is on the library path).
COMPLEX_EMOJI = features.check_feature("raqm")
NOTE_MIN = 2.0  # seconds a note stays up, even over fast cuts
STAMP_LEN = 2.6
FADE = 0.25


def style_for(look: str) -> str:
    return LOOK_STYLE.get(look, "minimal")


def _font(path: str, index: int, size: int):
    try:
        return ImageFont.truetype(path, size, index=index)
    except OSError:
        return ImageFont.truetype(f"{FONTS}/Helvetica.ttc", size)


def _is_emoji(ch: str) -> bool:
    o = ord(ch)
    return 0x1F000 <= o <= 0x1FAFF or 0x2600 <= o <= 0x27BF or 0x2B00 <= o <= 0x2BFF


def _runs(text: str):
    """Split text into (is_emoji, chunk) runs; flags (regional-indicator pairs) stay together."""
    if not COMPLEX_EMOJI:
        text = "".join(ch for ch in text if not 0x1F1E6 <= ord(ch) <= 0x1F1FF).replace("\u200d", "")
    runs: list[tuple[bool, str]] = []
    for ch in text:
        if ch in "️‍":
            if runs and runs[-1][0]:
                runs[-1] = (True, runs[-1][1] + ch)
            continue
        e = _is_emoji(ch)
        if runs and runs[-1][0] == e and not e:
            runs[-1] = (False, runs[-1][1] + ch)
        else:
            runs.append((e, ch))
    out: list[tuple[bool, str]] = []
    for e, chunk in runs:  # merge regional indicators into flags
        if e and out and out[-1][0] and len(out[-1][1]) == 1 and 0x1F1E6 <= ord(out[-1][1]) <= 0x1F1FF and 0x1F1E6 <= ord(chunk[0]) <= 0x1F1FF:
            out[-1] = (True, out[-1][1] + chunk)
        else:
            out.append((e, chunk))
    return out


def _line_image(text: str, font, size: int, fill, tracking: float = 0.0) -> Image.Image:
    """Render one line with emoji support (Apple Color Emoji only has a 160px strike)."""
    parts = []
    for is_e, chunk in _runs(text):
        if is_e:
            ef = ImageFont.truetype(EMOJI, 160)
            box = ef.getbbox(chunk)
            im = Image.new("RGBA", (max(1, box[2]) + 8, 200), (0, 0, 0, 0))
            ImageDraw.Draw(im).text((4, 0), chunk, font=ef, embedded_color=True)
            im = im.crop(im.getbbox() or (0, 0, 1, 1))
            h = int(size * 0.95)
            im = im.resize((max(1, int(im.width * h / max(1, im.height))), h), Image.LANCZOS)
            parts.append(("img", im))
        else:
            parts.append(("txt", chunk))
    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    widths = []
    for kind, p in parts:
        if kind == "img":
            widths.append(p.width + int(size * 0.12))
        else:
            widths.append(int(font.getlength(p) + tracking * size * len(p)))
    im = Image.new("RGBA", (max(1, sum(widths)), line_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    x = 0
    for (kind, p), w in zip(parts, widths):
        if kind == "img":
            im.alpha_composite(p, (x + int(size * 0.06), max(0, (line_h - p.height) // 2)))
        elif tracking:
            for ch in p:
                d.text((x, 0), ch, font=font, fill=fill)
                x += int(font.getlength(ch) + tracking * size)
            continue
        else:
            d.text((x, 0), p, font=font, fill=fill)
        x += w
    return im


def _measure(text: str, font) -> float:
    size = font.size
    return sum(size * 1.1 if e else font.getlength(chunk) for e, chunk in _runs(text))


def _wrap(text: str, font, max_w: float, max_lines: int = 2) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if _measure(trial, font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".,") + "…"
    return lines


def _layout(platform: str, W: int, H: int, letterbox: bool):
    """Return (left, top_y, bottom_y, max_width) for text, avoiding the app's UI and cinema bars."""
    safe = PLATFORMS.get(platform, {}).get("safe") or []
    top_pct = max([t + h for l, t, w, h in safe if t == 0 and w >= 90] or [0])
    bottom_pct = max([h for l, t, w, h in safe if t + h >= 99 and w >= 90] or [0])
    right_pct = max([w for l, t, w, h in safe if l + w >= 99 and w < 50] or [0])
    margin = 0.055 * min(W, H)
    bar = (H - W / 2.39) / 2 if letterbox and W / H >= 1.5 else 0
    left = margin
    top = H * top_pct / 100 + margin + bar
    bottom = H * (1 - bottom_pct / 100) - margin - bar
    max_w = W * (1 - right_pct / 100) - left - margin
    return left, top, bottom, max_w


def _shadowed(canvas: Image.Image, items: list[tuple[Image.Image, int, int]], size: int) -> Image.Image:
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    for im, x, y in items:
        sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
        sh.putalpha(im.getchannel("A").point(lambda a: int(a * 0.75)))
        shadow.alpha_composite(sh, (x + max(1, size // 30), y + max(2, size // 18)))
        canvas.alpha_composite(im, (x, y))
    return Image.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(size * 0.15)), canvas)


def note_png(text: str, style: str, platform: str, W: int, H: int, letterbox: bool, out: Path):
    """A note centred in the lower part of the frame, inside the area the app leaves free."""
    note_f, note_i, _, _, upper, pill = STYLES[style]
    size = int(min(W, H) * (0.056 if style == "serif" else 0.05))
    left, _, bottom, max_w = _layout(platform, W, H, letterbox)
    body = text.upper() if upper else text
    for scale in (1.0, 0.92, 0.85):  # shrink a little rather than strand a word or emoji on its own line
        size_s = int(size * scale)
        font = _font(note_f, note_i, size_s)
        pad = int(size_s * 0.5) if pill else 0
        lines = _wrap(body, font, max_w - 2 * pad)
        if len(lines) == 1 or len(lines[-1].split()) > 2:
            break
    size = size_s
    fill = (20, 20, 22, 255) if pill else (255, 255, 255, 255)
    imgs = [_line_image(line, font, size, fill) for line in lines]
    gap = int(size * 0.14)
    block_w = max(i.width for i in imgs)
    block_h = sum(i.height for i in imgs) + gap * (len(imgs) - 1)
    cx = left + max_w / 2  # centre of the free area (left of TikTok / Reels buttons)
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if pill:  # one rounded box around every line, not a pill per line
        box = Image.new("RGBA", (block_w + 2 * pad, block_h + pad), (0, 0, 0, 0))
        ImageDraw.Draw(box).rounded_rectangle([0, 0, box.width - 1, box.height - 1], radius=int(size * 0.35),
                                              fill=(255, 255, 255, 242))
        y = pad // 2
        for im in imgs:
            box.alpha_composite(im, ((box.width - im.width) // 2, y))
            y += im.height + gap
        canvas.alpha_composite(box, (int(cx - box.width / 2), int(bottom - box.height)))
    else:
        y = bottom - block_h
        items = []
        for im in imgs:
            items.append((im, int(cx - im.width / 2), int(y)))
            y += im.height + gap
        canvas = _shadowed(canvas, items, size)
    canvas.save(out)


def stamp_png(place: str, date: str, style: str, platform: str, W: int, H: int, letterbox: bool, out: Path):
    """A location / date tag in the top-left corner, under the app's top bar."""
    _, _, st_f, st_i, _, pill = STYLES[style]
    base = int(min(W, H) * 0.068)
    left, top, _, max_w = _layout(platform, W, H, letterbox)
    date_font, date_idx = (f"{SUPP}/Baskerville.ttc", 4) if style == "serif" else (f"{FONTS}/Avenir Next.ttc", 2)
    dsize = int(base * 0.5)
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    items, y = [], top
    if place:
        im = _line_image(place.upper(), _font(st_f, st_i, base), base, (255, 255, 255, 255), tracking=0.12)
        if im.width > max_w:
            im = im.resize((int(max_w), max(1, int(im.height * max_w / im.width))), Image.LANCZOS)
        if pill:
            pad = int(base * 0.35)
            box = Image.new("RGBA", (im.width + 2 * pad, im.height + pad), (0, 0, 0, 0))
            ImageDraw.Draw(box).rounded_rectangle([0, 0, box.width - 1, box.height - 1], radius=int(base * 0.3),
                                                  fill=(243, 168, 91, 245))
            box.alpha_composite(im, (pad, pad // 2))
            canvas.alpha_composite(box, (int(left), int(y)))
            y += box.height + base * 0.22
        else:
            items.append((im, int(left), int(y)))
            y += im.height
            if date:
                rule_y = y + base * 0.12
                ImageDraw.Draw(canvas).line([(left, rule_y), (left + base * 1.1, rule_y)],
                                            fill=(255, 255, 255, 220), width=max(2, W // 360))
                y = rule_y + base * 0.22
    if date:
        im = _line_image(date.upper(), _font(date_font, date_idx, dsize), dsize, (255, 255, 255, 235), tracking=0.16)
        items.append((im, int(left), int(y)))
    _shadowed(canvas, items, base).save(out)


def fmt_date(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%-d %b %Y")


def overlays(plan: dict, W: int, H: int, workdir: Path) -> list[tuple[Path, float, float]]:
    """PNG overlays with their (start, end) times on the final timeline."""
    text = plan.get("text") or {}
    shots = plan["shots"]
    total = plan["duration"]
    style = style_for(plan["look"])
    letterbox = plan["settings"].get("letterbox", True) and plan["look"] == "cinematic"
    out: list[tuple[Path, float, float]] = []
    title_end = 3.6 if (plan["settings"].get("title") or "").strip() else 0.0

    place, show_date = (text.get("place") or "").strip(), bool(text.get("show_date"))
    if place or show_date:
        last_day, last_end = None, -9.0
        for i, s in enumerate(shots):
            day = s["captured"][:10]
            start = max(s["timeline"][0], title_end if i == 0 or s["timeline"][0] < title_end else 0)
            if day != last_day and start - last_end > 1.0 and start < total - 1.5:
                end = min(total - 0.3, start + STAMP_LEN)
                png = workdir / f"stamp_{i}.png"
                stamp_png(place, fmt_date(s["captured"]) if show_date else "", style, plan["platform"], W, H, letterbox, png)
                out.append((png, start, end))
                last_end = end
            last_day = day

    notes = {int(k): v.strip() for k, v in (text.get("notes") or {}).items() if v and v.strip()}
    prev_end = -1.0
    for i in sorted(notes):  # queued: a note on a quick cut pushes the next one back instead of being dropped
        if i >= len(shots):
            continue
        a, b = shots[i]["timeline"]
        start = max(a + 0.1, prev_end + 0.15)
        end = min(max(b, start + NOTE_MIN), total - 0.2)
        if end - start < 0.6:
            continue
        prev_end = end
        png = workdir / f"note_{i}.png"
        note_png(notes[i], style, plan["platform"], W, H, letterbox, png)
        out.append((png, start, end))
    return out
