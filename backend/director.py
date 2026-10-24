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
TRANSITION_DUR = {"dissolve": 0.6, "dip": 0.7, "whip": 0.3, "zoom": 0.35, "flash": 0.25, "leak": 0.8, "graydip": 0.6,
                  # names used by edits made before the transition overhaul
                  "fade": 0.5, "fadeblack": 0.7, "smoothleft": 0.3, "hblur": 0.3}
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
    action_t: float | None = None  # clip time of the strongest action in this window
    action: float = 0.0

    @property
    def center_t(self) -> float:
        t = self.an["t"]
        return (t[self.i0] + t[self.i1 - 1]) / 2


def _rank(values: np.ndarray) -> np.ndarray:
    """Percentile rank in 0..1; equal values share the same (average) rank."""
    n = len(values)
    if n == 0:
        return values
    ranks = np.empty(n)
    ranks[values.argsort(kind="stable")] = np.arange(n)
    _, inv = np.unique(values, return_inverse=True)
    avg = np.bincount(inv, ranks) / np.bincount(inv)
    return avg[inv] / max(1, n - 1)


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
        zeros = np.zeros(n)
        smile = np.asarray(an.get("smile") or zeros, dtype=float)
        flat_frame = np.asarray(an.get("flat") or zeros, dtype=float)
        swing = np.hypot(np.asarray(an["dx"]), np.asarray(an["dy"]))
        raw_audio = np.asarray(an["audio"], dtype=float)
        heard = raw_audio[raw_audio > 0]
        event = np.zeros(n)
        if heard.size:
            event = np.clip((20 * np.log10((raw_audio + 1e-6) / (np.median(heard) + 1e-6)) - 6) / 10, 0, 1)
        act = np.asarray(an.get("act") or zeros, dtype=float)
        action = np.clip((act / (np.median(act) + 0.002) - 2) / 4, 0, 1) * (act > 0.004)
        action *= np.clip(1 - (jitter_abs - 0.01) * 60, 0, 1)  # shaky hands aren't action
        dur = next(c["duration"] for c in clips if c["id"] == cid)
        motion_pref = 1 - np.abs(motion - 0.6) / 0.6
        s = (0.31 * sharp + 0.23 * color + 0.14 * motion_pref + 0.13 * faces + 0.08 * smile
             + 0.06 * np.maximum(event, 0.5 * audio) + 0.12 * action)
        s -= np.clip((jitter_abs - 0.012) * 18, 0, 0.6)
        s -= np.where(bright < 0.1, 0.5, 0) + np.where(bright > 0.88, 0.4, 0) + np.clip(clip_hi - 0.15, 0, 1)
        s -= 0.6 * flat_frame + np.clip((swing - 0.12) * 4, 0, 0.5)  # pocket / ground shots, whip pans
        s -= np.where((t < EDGE_GUARD) | (t > dur - EDGE_GUARD), 1.0, 0)
        wow = 0.35 * color + 0.25 * sharp + 0.2 * motion + 0.2 * np.maximum(action, smile)
        out[cid] = {"score": s, "wow": wow, "calm": 1 - motion, "action": action}
    return out


def find_candidates(clip, an, sc, win: int, per_clip: int, use_smart: bool = False) -> list[Candidate]:
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
        c = Candidate(clip, an, int(i), int(i + win), float(val[i]),
                      float(sc["wow"][i:i + win].mean()), float(sc["calm"][i:i + win].mean()),
                      sig[i:i + win].mean(axis=0))
        act = sc["action"][i:i + win]
        j = int(np.argmax(act)) if len(act) else 0
        if len(act) and act[j] > 0.25:
            c.action_t, c.action = float(an["t"][i + j]), float(act[j])
        if clip.get("starred"):  # guaranteed a place elsewhere; only a nudge here so it doesn't take over
            c.score += 0.08
            c.wow += 0.1
        if (smart := clip.get("smart")) and use_smart:
            c.score += 0.3 * (smart["interest"] - 5) / 5 - (0.15 if smart["label"] == "boring" else 0)
            c.wow += 0.2 * (smart["interest"] - 5) / 5
        cands.append(c)
    return cands


def merge_slots(slots: list[dict], cap: int) -> list[dict]:
    """Join neighbouring beat slots until there are at most `cap`, never across the drop or a section change."""
    slots = [dict(s) for s in slots]
    while len(slots) > cap:
        joinable = [k for k in range(len(slots) - 1)
                    if not slots[k + 1]["drop"] and not slots[k + 1]["section_start"]]
        if not joinable:
            break
        k = min(joinable, key=lambda k: slots[k + 1]["t1"] - slots[k]["t0"])
        a, b = slots[k], slots[k + 1]
        slots[k] = {**a, "t1": b["t1"], "energy": round((a["energy"] + b["energy"]) / 2, 3), "pre_drop": b["pre_drop"]}
        del slots[k + 1]
    return slots


def shot_lengths(target: float, look: Look, rng, stretch: float = 1.0) -> list[tuple[float, float]]:
    """Timeline boundaries [(start, end), ...] for each shot when there's no music to follow."""
    lo, hi = look.pace[0] * stretch, look.pace[1] * stretch
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


def choose_crop(clip: dict, an: dict, i0: int, i1: int, out_w: int, out_h: int, rng, still: bool = False) -> dict:
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
    if axis == "x" and not concentrated and not still:
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
    if day_change and "dip" in look.transitions:
        return "dip"
    names, weights = zip(*look.transitions.items())
    return str(rng.choice(names, p=np.asarray(weights) / sum(weights)))


def arrange_to_music(chosen: list[Candidate], slots: list[dict], hook: bool) -> list[Candidate]:
    """Best shot on the drop, a hook up front, then story order with shot motion matched to music energy."""
    n = min(len(slots), len(chosen))
    remaining = list(chosen)
    out: list[Candidate | None] = [None] * n
    by_wow = lambda c: c.wow + 0.5 * c.score + 0.3 * c.action  # noqa: E731
    drop_k = next((k for k, s in enumerate(slots[:n]) if s["drop"]), None)
    if drop_k is not None:  # a real action moment is the best thing to hit the drop with
        out[drop_k] = max(remaining, key=lambda c: by_wow(c) + 0.5 * c.action)
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


def music_transition(look: Look, rng, slot: dict, prev_slot: dict | None = None) -> str:
    if slot["drop"] or (prev_slot and prev_slot.get("pre_drop")):
        return "cut"
    fancy = {k: v for k, v in look.transitions.items() if k != "cut"}
    names, weights = zip(*fancy.items()) if fancy else ((), ())
    if slot["section_start"] and fancy:
        return str(rng.choice(names, p=np.asarray(weights) / sum(weights)))
    share = 1 - look.transitions.get("cut", 1.0)
    if slot.get("phrase") and fancy and rng.random() < min(0.9, share * 1.8):  # every 4 bars: a breathing point
        return str(rng.choice(names, p=np.asarray(weights) / sum(weights)))
    if fancy and look.energetic and slot["energy"] > 0.7 and rng.random() < 0.15:
        return "whip" if "whip" in fancy else next(iter(fancy))
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


def src_offset(t: float, speed: float, ramp, out_len: float) -> float:
    """Clip seconds played by output time t (constant speed, or a linear speed ramp)."""
    if ramp and abs(ramp[1] - ramp[0]) > 1e-3:
        a, b = ramp
        return a * t + (b - a) / (2 * out_len) * t * t
    return t * speed


def out_time(s: float, speed: float, ramp, out_len: float) -> float:
    if ramp and abs(ramp[1] - ramp[0]) > 1e-3:
        a, b = ramp
        k = (b - a) / (2 * out_len)
        return (-a + math.sqrt(max(0.0, a * a + 4 * k * s))) / (2 * k)
    return s / speed


def action_target(out_len: float, head: float, a: float, b: float, slot: dict | None,
                  beats: list[float] | None, bars: list[float] | None) -> float:
    """Where (seconds into this shot's output) its action should land."""
    if slot and slot["drop"]:
        return head + 0.08  # the action hits with the drop
    lo, hi = head + min(0.45, 0.3 * (b - a)), out_len - 0.3
    want = 0.4 * out_len  # leave time to see the build-up, and the aftermath
    for grid in (bars, beats):
        pts = [t - a + head for t in (grid or []) if lo <= t - a + head <= hi]
        if pts:
            return min(pts, key=lambda p: abs(p - want))
    return float(np.clip(want, min(0.5, out_len / 2), max(0.5, out_len - 0.4)))


def plan_edit(clips: list[dict], analyses: dict[str, dict], settings: dict, slots: list[dict] | None = None,
              beats: list[float] | None = None, bars: list[float] | None = None, refine=None) -> dict:
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

    # Few clips: make fewer, longer shots rather than showing the same clips over and over.
    cap = max(4, 2 * len(clips) + 1)
    if slots:
        slots = merge_slots(slots, cap)
        bounds = [(s["t0"], s["t1"]) for s in slots]
    else:
        stretch = 1.0
        bounds = shot_lengths(target, look, rng)
        while len(bounds) > cap and stretch < 3:
            stretch *= 1.2
            bounds = shot_lengths(target, look, np.random.default_rng(seed), stretch)
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
        pool += find_candidates(c, an, sc[c["id"]], win, per_clip, bool(settings.get("smart_picks")))
    if not pool:
        raise ValueError("Couldn't find usable moments (everything was too dark, shaky or short)")
    if seed:
        for cand in pool:
            cand.score += float(rng.normal(0, 0.06))

    chosen: list[Candidate] = []
    use_count: dict[str, int] = {}
    max_per_clip = max(1, math.ceil(n_shots / len(clips)))
    remaining = sorted(pool, key=lambda x: -x.score)
    for c in sorted((c for c in clips if c.get("starred")), key=lambda c: c["captured"]):
        mine = [x for x in remaining if x.clip["id"] == c["id"]]
        if mine and len(chosen) < n_shots:  # starred clips are always in the edit
            best = max(mine, key=lambda x: x.score)
            chosen.append(best)
            remaining.remove(best)
            use_count[c["id"]] = 1
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
            trans[k] = music_transition(look, rng, slots[k], slots[k - 1])
        else:
            trans[k] = pick_transition(look, rng, k, chosen[k - 1].clip, chosen[k].clip)
    tdur = [TRANSITION_DUR.get(t, 0.0) for t in trans]
    for k in range(1, len(chosen)):
        shortest = min(bounds[k - 1][1] - bounds[k - 1][0], bounds[k][1] - bounds[k][0])
        if tdur[k] > 0.45 * shortest:
            tdur[k] = round(0.45 * shortest, 3) if shortest > 0.6 else 0.0
            if tdur[k] == 0.0:
                trans[k] = "cut"

    whip_dir = {}
    for k in range(1, len(chosen)):
        if trans[k] == "whip":  # keep the camera's own movement going through the whip
            prev = chosen[k - 1]
            dx = float(np.mean(prev.an["dx"][prev.i0:prev.i1] or [0]))
            dy = float(np.mean(prev.an["dy"][prev.i0:prev.i1] or [0]))
            if max(abs(dx), abs(dy)) < 0.002:
                whip_dir[k] = "left" if k % 2 else "right"
            elif abs(dx) >= abs(dy):
                whip_dir[k] = "left" if dx < 0 else "right"
            else:
                whip_dir[k] = "up" if dy < 0 else "down"

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
        lo_start, hi_start = 0.3, max(0.3, clip["duration"] - 0.3 - need)
        start = float(np.clip(center - need / 2, lo_start, hi_start))
        action = None
        if cand.action_t is not None:
            peak, strength = cand.action_t, cand.action
            if refine:
                found = refine(clip, peak - 1.0, peak + 1.0)
                if found:
                    peak, strength = found[0], max(strength, min(1.0, found[1] / 6))
            target = action_target(out_len, head, a, b, slot, beats, bars)
            start = float(np.clip(peak - src_offset(target, speed, fx["ramp"], out_len), lo_start, hi_start))
            landed = out_time(peak - start, speed, fx["ramp"], out_len)
            if 0 <= landed <= out_len:
                action = {"t": round(a - head + landed, 3), "strength": round(float(strength), 2)}
        f0, f1 = cand.i0, cand.i1
        if cand.action_t is not None:  # aim the frame at where the action happens, not the shot's average
            j = int(np.argmin(np.abs(np.asarray(an["t"]) - cand.action_t)))
            f0, f1 = max(0, j - 1), min(len(an["t"]), j + 2)
        crop = choose_crop(clip, an, f0, f1, out_w, out_h, rng, still=cand.action_t is not None)
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
            "transition_in": {"type": trans[k], "dur": tdur[k], **({"dir": whip_dir[k]} if k in whip_dir else {})},
            "blur_in": {"dir": whip_dir[k], "dur": round(tdur[k] / 2 + 0.06, 3)} if k in whip_dir else None,
            "blur_out": ({"dir": whip_dir[k + 1], "dur": round(tdur[k + 1] / 2 + 0.06, 3)}
                         if k + 1 in whip_dir else None),
            "crop": crop,
            "ramp": fx["ramp"],
            "punch": fx["punch"],
            "flash": fx["flash"],
            "music": {k2: slot[k2] for k2 in ("energy", "drop", "section_start", "pre_drop")} if slot else None,
            "action": action,
            "starred": bool(clip.get("starred")),
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
