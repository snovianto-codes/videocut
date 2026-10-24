"""The auto-editor: turns analysed clips into an edit decision list (shots, timing, crops, transitions)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .looks import LOOKS, Look

# safe: areas each app covers with its own buttons and captions, as [left, top, width, height] in % of the frame
PLATFORMS = {
    "reels": {"name": "Instagram Reels", "aspect": "9:16", "w": 1080, "h": 1920,
              "safe": [[0, 0, 100, 13], [0, 80, 100, 20], [86, 45, 14, 35]]},
    "tiktok": {"name": "TikTok", "aspect": "9:16", "w": 1080, "h": 1920,
               "safe": [[0, 0, 100, 10], [0, 76, 100, 24], [84, 35, 16, 41]]},
    "shorts": {"name": "YouTube Shorts", "aspect": "9:16", "w": 1080, "h": 1920,
               "safe": [[0, 0, 100, 10], [0, 80, 100, 20], [85, 45, 15, 35]]},
    "feed": {"name": "Instagram feed", "aspect": "4:5", "w": 1080, "h": 1350},
    "square": {"name": "Square post", "aspect": "1:1", "w": 1080, "h": 1080},
    "youtube": {"name": "YouTube 1080p", "aspect": "16:9", "w": 1920, "h": 1080},
    "youtube4k": {"name": "YouTube 4K", "aspect": "16:9", "w": 3840, "h": 2160},
    # edits made before Reels / TikTok / Shorts were split
    "vertical": {"name": "Vertical (Reels · TikTok · Shorts)", "aspect": "9:16", "w": 1080, "h": 1920, "hidden": True,
                 "safe": [[0, 0, 100, 13], [0, 78, 100, 22], [85, 40, 15, 38]]},
}
TRANSITION_DUR = {"fade": 0.5, "fadeblack": 0.7, "smoothleft": 0.3, "hblur": 0.3}
EDGE_GUARD = 0.6


@dataclass
class Candidate:
    clip: dict
    an: dict
    i0: int
    i1: int
    score: float
    wow: float
    calm: float
    sig: np.ndarray

    @property
    def center_t(self) -> float:
        t = self.an["t"]
        return (t[self.i0] + t[self.i1 - 1]) / 2


def _rank(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    order = values.argsort().argsort()
    return order / max(1, len(values) - 1)


def score_samples(clips: list[dict], analyses: dict[str, dict]) -> dict[str, dict[str, np.ndarray]]:
    """Group-normalised per-sample scores for each clip."""
    keys = [c["id"] for c in clips if c["id"] in analyses and analyses[c["id"]]["t"]]
    flat = {k: np.concatenate([np.asarray(analyses[cid][k], dtype=float) for cid in keys]) for k in ("sharp", "color", "motion", "audio")}
    jit = []
    for cid in keys:
        dx, dy = np.asarray(analyses[cid]["dx"]), np.asarray(analyses[cid]["dy"])
        j = np.zeros_like(dx)
        if len(dx) > 2:
            j[1:] = np.hypot(np.diff(dx), np.diff(dy))
        jit.append(j)
    flat["jitter"] = np.concatenate(jit) if jit else np.zeros(0)
    ranks = {
        "sharp": _rank(np.log1p(flat["sharp"])),
        "color": _rank(flat["color"]),
        "motion": _rank(flat["motion"]),
        "audio": _rank(flat["audio"]),
        "jitter": _rank(flat["jitter"]),
    }
    out, pos = {}, 0
    for idx, cid in enumerate(keys):
        an = analyses[cid]
        n = len(an["t"])
        sl = slice(pos, pos + n)
        pos += n
        sharp, color, motion, audio = ranks["sharp"][sl], ranks["color"][sl], ranks["motion"][sl], ranks["audio"][sl]
        jitter_abs = jit[idx]
        bright = np.asarray(an["bright"])
        clip_hi = np.asarray(an["clip_hi"])
        faces = np.asarray(an["faces"]) > 0
        t = np.asarray(an["t"])
        dur = next(c["duration"] for c in clips if c["id"] == cid)
        motion_pref = 1 - np.abs(motion - 0.6) / 0.6
        s = 0.34 * sharp + 0.26 * color + 0.18 * motion_pref + 0.06 * audio + 0.16 * faces
        s -= np.clip((jitter_abs - 0.012) * 18, 0, 0.6)
        s -= np.where(bright < 0.1, 0.5, 0) + np.where(bright > 0.88, 0.4, 0) + np.clip(clip_hi - 0.15, 0, 1)
        s -= np.where((t < EDGE_GUARD) | (t > dur - EDGE_GUARD), 1.0, 0)
        wow = 0.4 * color + 0.3 * sharp + 0.3 * motion
        out[cid] = {"score": s, "wow": wow, "calm": 1 - motion}
    return out


def find_candidates(clip, an, sc, win: int, per_clip: int) -> list[Candidate]:
    s = sc["score"]
    n = len(s)
    if n == 0:
        return []
    win = max(1, min(win, n))
    kernel = np.ones(win) / win
    mean = np.convolve(s, kernel, mode="valid")
    low = np.array([s[i:i + win].min() for i in range(len(mean))])
    val = mean + 0.3 * low
    cands, taken = [], np.zeros(n, bool)
    sig = np.asarray(an["sig"], dtype=float)
    for i in np.argsort(-val):
        if len(cands) >= per_clip:
            break
        if taken[i:i + win].any() or val[i] < -0.2:
            continue
        taken[max(0, i - win // 2):i + win + win // 2] = True
        cands.append(Candidate(clip, an, int(i), int(i + win), float(val[i]),
                               float(sc["wow"][i:i + win].mean()), float(sc["calm"][i:i + win].mean()),
                               sig[i:i + win].mean(axis=0)))
    return cands


def shot_lengths(target: float, look: Look, rng) -> list[tuple[float, float]]:
    """Timeline boundaries [(start, end), ...] for each shot when there's no music to follow."""
    lo, hi = look.pace
    bounds, t, shot = [0.0], 0.0, 0
    while t < target - lo * 0.6:
        length = lo * 0.75 if shot == 0 else rng.uniform(lo, hi)
        t = min(target, t + length)
        bounds.append(t)
        shot += 1
    if len(bounds) >= 3 and bounds[-1] - bounds[-2] < lo * 0.6:
        bounds.pop(-2)
    bounds[-1] = target
    return list(zip(bounds, bounds[1:]))


def base_speed(clip: dict, look: Look, opts: dict, has_faces_talking: bool) -> float:
    if not (look.slowmo and opts.get("slowmo", True)) or has_faces_talking:
        return 1.0
    fps = clip.get("fps", 30)
    if fps >= 100:
        return 0.4
    if fps >= 48 and look.fps <= 30 and (fps >= 2 * look.fps * 0.95):
        return 0.5
    return 1.0


def choose_crop(clip: dict, an: dict, i0: int, i1: int, out_w: int, out_h: int, rng) -> dict:
    sw, sh = clip["width"], clip["height"]
    a_src, a_out = sw / sh, out_w / out_h
    if abs(a_src - a_out) < 0.02:
        return {"mode": "fit"}
    if a_src > a_out:
        frac = a_out / a_src
        prof = np.asarray(an["colx"][i0:i1], dtype=float).mean(axis=0)
        fx = [v for v in an["face_x"][i0:i1] if v is not None]
        axis = "x"
    else:
        frac = a_src / a_out
        if frac < 0.6:
            return {"mode": "blurfill"}
        prof = np.asarray(an["coly"][i0:i1], dtype=float).mean(axis=0)
        fx = [v for v in an["face_y"][i0:i1] if v is not None]
        axis = "y"
    if fx:
        c = float(np.median(fx))
        return {"mode": "crop", "axis": axis, "frac": frac, "pos0": _pos(c, frac), "pos1": _pos(c, frac), "why": "face"}
    bins = len(prof)
    width = max(1, int(round(frac * bins)))
    centers = np.arange(bins - width + 1)
    prior = np.exp(-(((centers + width / 2) / bins - 0.5) ** 2) / 0.08)
    capt = np.array([prof[i:i + width].sum() for i in centers]) * (0.6 + 0.4 * prior)
    best = int(np.argmax(capt))
    c = (best + width / 2) / bins
    concentrated = prof[best:best + width].sum() > frac * 1.35
    if axis == "x" and not concentrated:
        dxs = np.asarray(an["dx"][i0:i1])
        direction = -np.sign(dxs.mean()) if abs(dxs.mean()) > 0.002 else rng.choice([-1, 1])
        slack = (1 - frac) * 0.35
        return {"mode": "crop", "axis": axis, "frac": frac,
                "pos0": _pos(0.5 - direction * slack, frac), "pos1": _pos(0.5 + direction * slack, frac), "why": "pan"}
    return {"mode": "crop", "axis": axis, "frac": frac, "pos0": _pos(c, frac), "pos1": _pos(c, frac), "why": "subject"}


def _pos(center: float, frac: float) -> float:
    """Crop offset as a fraction of the slack (0 = left/top, 1 = right/bottom)."""
    if frac >= 1:
        return 0.5
    return float(np.clip((center - frac / 2) / (1 - frac), 0, 1))


def pick_transition(look: Look, rng, k: int, prev_clip: dict, clip: dict) -> str:
    if k == 1:
        return "cut"
    day_change = prev_clip["captured"][:10] != clip["captured"][:10]
    if day_change and "fadeblack" in look.transitions:
        return "fadeblack"
    names, weights = zip(*look.transitions.items())
    return str(rng.choice(names, p=np.asarray(weights) / sum(weights)))


def arrange_to_music(chosen: list[Candidate], slots: list[dict], hook: bool) -> list[Candidate]:
    """Best shot on the drop, a hook up front, then story order with shot motion matched to music energy."""
    n = min(len(slots), len(chosen))
    remaining = list(chosen)
    out: list[Candidate | None] = [None] * n
    by_wow = lambda c: c.wow + 0.5 * c.score  # noqa: E731
    drop_k = next((k for k, s in enumerate(slots[:n]) if s["drop"]), None)
    if drop_k is not None:
        out[drop_k] = max(remaining, key=by_wow)
        remaining.remove(out[drop_k])
    if hook and out[0] is None and remaining:
        out[0] = max(remaining, key=by_wow)
        remaining.remove(out[0])
    motion = np.array([1 - c.calm for c in chosen])
    rank = {id(c): r for c, r in zip(chosen, _rank(motion))}
    for k in range(n):
        if out[k] is not None or not remaining:
            continue
        opts = remaining[:4]
        neighbours = {o.clip["id"] for o in (out[k - 1] if k else None, out[k + 1] if k + 1 < n else None) if o}
        pick = min(opts, key=lambda c: abs(rank[id(c)] - slots[k]["energy"]) + 0.08 * opts.index(c)
                   + (0.6 if c.clip["id"] in neighbours else 0))
        out[k] = pick
        remaining.remove(pick)
    return [c for c in out if c is not None]


def music_transition(look: Look, rng, slot: dict) -> str:
    if slot["drop"]:
        return "cut"
    fancy = {k: v for k, v in look.transitions.items() if k != "cut"}
    if slot["section_start"] and fancy:
        names, weights = zip(*fancy.items())
        return str(rng.choice(names, p=np.asarray(weights) / sum(weights)))
    if fancy and look.energetic and slot["energy"] > 0.7 and rng.random() < 0.15:
        return "smoothleft" if "smoothleft" in fancy else next(iter(fancy))
    return "cut"


def shot_effects(look: Look, slot: dict, clip: dict, settings: dict) -> dict:
    fx = {"ramp": None, "punch": 0.0, "flash": False}
    if not settings.get("ramps", True):
        return fx
    hi_fps = clip.get("fps", 30) >= 48
    if slot["pre_drop"]:
        slow = 0.45 if hi_fps else 0.65
        fx["ramp"] = [1.5 if look.energetic else 1.0, slow]
    if slot["drop"]:
        fx["punch"] = look.punch
        fx["flash"] = look.flash
        if look.energetic:
            fx["ramp"] = [1.8, 1.0]
    elif slot["section_start"] and look.energetic and slot["energy"] > 0.6:
        fx["punch"] = round(look.punch * 0.6, 3)
    return fx


def plan_edit(clips: list[dict], analyses: dict[str, dict], settings: dict, slots: list[dict] | None = None) -> dict:
    look = LOOKS[settings.get("look", "cinematic")]
    plat = PLATFORMS[settings.get("platform", "reels")]
    target = float(settings.get("duration", 20))
    seed = int(settings.get("seed", 0))
    rng = np.random.default_rng(seed)
    out_w, out_h = plat["w"], plat["h"]
    excluded = set(settings.get("exclude_clips", []))
    clips = [c for c in clips if c["id"] in analyses and c["id"] not in excluded]
    if not clips:
        raise ValueError("No analysed clips in this group")

    bounds = [(s["t0"], s["t1"]) for s in slots] if slots else shot_lengths(target, look, rng)
    trans: list[str] = ["cut"] * len(bounds)
    n_shots = len(bounds)

    sc = score_samples(clips, analyses)
    med_len = float(np.median([b - a for a, b in bounds]))
    per_clip = max(2, math.ceil(n_shots / max(1, len(clips))) + 1)
    pool: list[Candidate] = []
    for c in clips:
        an = analyses[c["id"]]
        sp = base_speed(c, look, settings, False)
        win = max(1, int(round(med_len * sp * an["fps"])))
        pool += find_candidates(c, an, sc[c["id"]], win, per_clip)
    if not pool:
        raise ValueError("Couldn't find usable moments (everything was too dark, shaky or short)")
    if seed:
        for cand in pool:
            cand.score += float(rng.normal(0, 0.06))

    chosen: list[Candidate] = []
    use_count: dict[str, int] = {}
    max_per_clip = max(1, math.ceil(n_shots / len(clips)))
    remaining = sorted(pool, key=lambda x: -x.score)
    while len(chosen) < n_shots and remaining:
        best, best_val = None, -9.0
        for cand in remaining:
            v = cand.score
            cid = cand.clip["id"]
            if use_count.get(cid, 0) >= max_per_clip:
                v -= 0.5 * use_count[cid]
            for ch in chosen:
                d = float(np.abs(ch.sig - cand.sig).mean())
                if d < 0.035:
                    v -= 0.35
                if ch.clip["id"] == cid and abs(ch.center_t - cand.center_t) < 4:
                    v -= 0.4
            if v > best_val:
                best, best_val = cand, v
        chosen.append(best)
        remaining.remove(best)
        use_count[best.clip["id"]] = use_count.get(best.clip["id"], 0) + 1
        if len(remaining) == 0 and len(chosen) < n_shots:
            remaining = [c for c in pool if c not in chosen] or list(pool)

    chosen.sort(key=lambda c: (c.clip["captured"], c.center_t))
    if slots:
        chosen = arrange_to_music(chosen, slots, settings.get("hook", True))
    else:
        if settings.get("hook", True) and len(chosen) > 2:
            hook = max(chosen, key=lambda c: c.wow + 0.5 * c.score)
            chosen.remove(hook)
            chosen.insert(0, hook)
        if len(chosen) > 3:
            tail = chosen[len(chosen) * 2 // 3:]
            ender = max(tail, key=lambda c: c.calm * 0.6 + c.score)
            chosen.remove(ender)
            chosen.append(ender)

    for k in range(1, len(chosen)):
        if slots:
            trans[k] = music_transition(look, rng, slots[k])
        else:
            trans[k] = pick_transition(look, rng, k, chosen[k - 1].clip, chosen[k].clip)
    tdur = [TRANSITION_DUR.get(t, 0.0) for t in trans]
    for k in range(1, len(chosen)):
        shortest = min(bounds[k - 1][1] - bounds[k - 1][0], bounds[k][1] - bounds[k][0])
        if tdur[k] > 0.45 * shortest:
            tdur[k] = round(0.45 * shortest, 3) if shortest > 0.6 else 0.0
            if tdur[k] == 0.0:
                trans[k] = "cut"

    shots = []
    for k, cand in enumerate(chosen):
        a, b = bounds[k]
        head = tdur[k] / 2
        tail = tdur[k + 1] / 2 if k + 1 < len(chosen) else 0.0
        if k == len(chosen) - 1:
            tail = 0.0
        out_len = (b - a) + head + tail
        clip, an = cand.clip, cand.an
        talking = (np.asarray(an["faces"][cand.i0:cand.i1]) > 0).mean() > 0.5 and np.mean(sc[clip["id"]]["score"]) > -1
        speed = base_speed(clip, look, settings, talking and bool(clip.get("has_audio")) and not settings.get("music"))
        slot = slots[k] if slots else None
        fx = {"ramp": None, "punch": 0.0, "flash": False}
        if slot:
            calm_music = slot["energy"] < 0.6 or (look.key in ("cinematic", "golden", "film") and not slot["drop"])
            if not calm_music:
                speed = 1.0
            fx = shot_effects(look, slot, clip, settings)
            if fx["ramp"]:
                speed = (fx["ramp"][0] + fx["ramp"][1]) / 2
        avail = clip["duration"] - 2 * 0.3
        need = out_len * speed
        if need > avail:
            if fx["ramp"]:
                fx["ramp"] = None
                speed = min(1.0, avail / out_len)
            else:
                speed = avail / out_len
            need = out_len * speed
        center = cand.center_t
        start = float(np.clip(center - need / 2, 0.3, max(0.3, clip["duration"] - 0.3 - need)))
        crop = choose_crop(clip, an, cand.i0, cand.i1, out_w, out_h, rng)
        shots.append({
            "clip_id": clip["id"],
            "rotate_fix": clip.get("rotate_fix", 0),
            "name": clip["name"],
            "path": clip["path"],
            "profile": clip["profile"],
            "ev": an.get("ev", 0.0) + float(settings.get("ev_bias", 0.0)),
            "start": round(start, 3),
            "src_dur": round(need, 3),
            "out_len": round(out_len, 3),
            "speed": round(speed, 3),
            "timeline": [round(a, 3), round(b, 3)],
            "transition_in": {"type": trans[k], "dur": tdur[k]},
            "crop": crop,
            "ramp": fx["ramp"],
            "punch": fx["punch"],
            "flash": fx["flash"],
            "music": {k2: slot[k2] for k2 in ("energy", "drop", "section_start", "pre_drop")} if slot else None,
            "score": round(cand.score, 3),
            "has_audio": clip.get("has_audio", False),
            "captured": clip["captured"],
            "camera": clip["camera"],
        })

    total = bounds[len(chosen) - 1][1]
    return {
        "settings": settings,
        "look": look.key,
        "platform": settings.get("platform", "reels"),
        "width": out_w,
        "height": out_h,
        "fps": look.fps,
        "duration": round(total, 3),
        "shots": shots,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
