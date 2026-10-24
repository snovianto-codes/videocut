"""Colour pipeline: camera log -> Rec.709 display, then a creative grade, baked into 3D LUTs."""
from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from .config import DATA_DIR, LUT_DIR

LUT_CACHE = DATA_DIR / "luts"
LUT_CACHE.mkdir(parents=True, exist_ok=True)
LUT_SIZE = 33

# ---------------------------------------------------------------- transfer functions
# Inputs are ffmpeg RGB in [0,1] after a limited-range YUV->RGB conversion.


def _code(rgb):
    """Legal-range normalised RGB back to 10-bit code value / 1023."""
    return (rgb * 876.0 + 64.0) / 1023.0


def slog3_to_lin(rgb):
    cv = _code(rgb) * 1023.0
    return np.where(
        cv >= 171.2102946929,
        (10.0 ** ((cv - 420.0) / 261.5)) * 0.19 - 0.01,
        (cv - 95.0) * 0.01125 / (171.2102946929 - 95.0),
    )


def slog2_to_lin(rgb):
    x = (10.0 ** ((rgb - 0.646596) / 0.432699) - 0.037584) * 219.0 / 155.0
    return x * 0.9


def flog_to_lin(rgb):
    a, b, c, d, e, f = 0.555556, 0.009468, 0.344676, 0.790453, 8.735631, 0.092864
    y = _code(rgb)
    return np.where(y >= 0.100537775223865, (10.0 ** ((y - d) / c) - b) / a, (y - f) / e)


def flog2_to_lin(rgb):
    a, b, c, d, e, f = 5.555556, 0.064829, 0.245281, 0.384316, 8.799461, 0.092864
    y = _code(rgb)
    return np.where(y >= 0.100686685370811, (10.0 ** ((y - d) / c) - b) / a, (y - f) / e)


def dlog_to_lin(rgb):
    y = _code(rgb)
    return np.where(y <= 0.14, (y - 0.0929) / 6.025, (10.0 ** (3.89616 * y - 2.27752) - 0.0108) / 0.9892)


def hlg_to_lin(rgb):
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    e = np.where(rgb <= 0.5, rgb ** 2 / 3.0, (np.exp((rgb - c) / a) + b) / 12.0)
    return e / 0.2647  # reference white (75% signal) -> 1.0, 18% grey (38%) -> ~0.18


# ---------------------------------------------------------------- gamuts
D65 = (0.3127, 0.3290)
PRIMARIES = {
    "rec709": ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06)),
    "rec2020": ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046)),
    "sgamut3cine": ((0.766, 0.275), (0.225, 0.800), (0.089, -0.087)),
    "sgamut": ((0.73, 0.28), (0.14, 0.855), (0.10, -0.05)),
    "dgamut": ((0.71, 0.31), (0.21, 0.88), (0.09, -0.08)),
}


def _rgb_to_xyz(prim):
    xyz = np.array([[x / y, 1.0, (1 - x - y) / y] for x, y in prim]).T
    wx, wy = D65
    w = np.array([wx / wy, 1.0, (1 - wx - wy) / wy])
    s = np.linalg.solve(xyz, w)
    return xyz * s


def gamut_matrix(src: str, dst: str = "rec709"):
    return np.linalg.inv(_rgb_to_xyz(PRIMARIES[dst])) @ _rgb_to_xyz(PRIMARIES[src])


PROFILES = {
    "slog3": (slog3_to_lin, "sgamut3cine", "Sony S-Log3"),
    "slog2": (slog2_to_lin, "sgamut", "Sony S-Log2"),
    "flog": (flog_to_lin, "rec2020", "Fujifilm F-Log"),
    "flog2": (flog2_to_lin, "rec2020", "Fujifilm F-Log2"),
    "dlog": (dlog_to_lin, "dgamut", "DJI D-Log"),
    "dlogm": (dlog_to_lin, "rec709", "DJI D-Log M"),
    "hlg": (hlg_to_lin, "rec2020", "HLG"),
    "rec709": (None, "rec709", "Standard (Rec.709)"),
}


def aces_fit(x):
    x = x * 0.6
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


def softclip(x, knee=0.8):
    over = np.maximum(x - knee, 0)
    return np.where(x > knee, knee + (1 - knee) * (1 - np.exp(-over / (1 - knee))), x)


# ---------------------------------------------------------------- official LUT support
def load_cube(path: Path):
    size, rows = None, []
    for line in path.read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.upper().startswith("LUT_3D_SIZE"):
            size = int(s.split()[1])
        elif s[0].isdigit() or s[0] in "-.":
            parts = s.split()
            if len(parts) == 3:
                rows.append([float(p) for p in parts])
    arr = np.array(rows, dtype=np.float32)
    return arr.reshape(size, size, size, 3)  # [b][g][r] order in .cube files


def apply_cube(cube, rgb):
    n = cube.shape[0]
    p = np.clip(rgb, 0, 1) * (n - 1)
    i0 = np.floor(p).astype(int).clip(0, n - 2)
    f = p - i0
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
    out = 0
    for db in (0, 1):
        for dg in (0, 1):
            for dr in (0, 1):
                w = (fr if dr else 1 - fr) * (fg if dg else 1 - fg) * (fb if db else 1 - fb)
                out = out + w * cube[b0 + db, g0 + dg, r0 + dr]
    return out


_official: dict[str, object] = {}


def official_lut(profile: str):
    if profile not in _official:
        p = LUT_DIR / f"{profile}.cube"
        _official[profile] = load_cube(p) if p.exists() else None
    return _official[profile]


def to_display(rgb, profile: str, ev: float = 0.0):
    """Camera RGB -> Rec.709 display-referred RGB, with exposure in stops."""
    cube = official_lut(profile) if profile != "rec709" else None
    decode, gamut, _ = PROFILES.get(profile, PROFILES["rec709"])
    if cube is not None or decode is None:
        base = apply_cube(cube, rgb) if cube is not None else rgb
        if abs(ev) < 1e-3:
            return np.clip(base, 0, 1)
        lin = np.clip(base, 0, 1) ** 2.2 * (2.0 ** ev)
        return np.clip(softclip(lin), 0, 1) ** (1 / 2.2)
    lin = decode(rgb)
    if gamut != "rec709":
        lin = lin @ gamut_matrix(gamut).T
    lin = np.maximum(lin, 0) * (2.0 ** ev)
    return aces_fit(lin) ** (1 / 2.2)


# ---------------------------------------------------------------- creative grade
@dataclass
class Look:
    key: str
    name: str
    blurb: str
    contrast: float = 0.1
    saturation: float = 1.0
    vibrance: float = 0.0
    warmth: float = 0.0
    shadows: tuple = (0.0, 0.0, 0.0)
    highlights: tuple = (0.0, 0.0, 0.0)
    lift: float = 0.0
    gain: float = 1.0
    teal_greens: float = 0.0
    mono: bool = False
    exposure_target: float = 0.42
    # render-side extras
    fps: int = 30
    vignette: float = 0.0
    grain: int = 0
    glow: float = 0.0
    letterbox: bool = False
    pace: tuple = (1.8, 2.8)
    transitions: dict = field(default_factory=lambda: {"cut": 0.8, "fade": 0.2})
    slowmo: bool = True
    punch: float = 0.06  # zoom-in amount that snaps back on the drop
    flash: bool = False  # white flash frame on the drop
    energetic: bool = False  # beat-level stutter cuts before drops, speed ramps out of them


LOOKS: dict[str, Look] = {
    "cinematic": Look(
        "cinematic", "Cinematic", "Teal & orange, soft contrast, 24 fps, widescreen bars, slow-mo on high-fps clips.",
        contrast=0.28, saturation=0.92, vibrance=0.1, warmth=0.02,
        shadows=(-0.025, 0.012, 0.035), highlights=(0.035, 0.012, -0.03), lift=0.015,
        teal_greens=0.6, exposure_target=0.40, fps=24, vignette=0.35, grain=4, letterbox=True, punch=0.05,
        pace=(2.2, 3.6), transitions={"cut": 0.75, "fade": 0.15, "fadeblack": 0.1},
    ),
    "vibrant": Look(
        "vibrant", "Travel Vibes", "Punchy colour, bright and warm, fast beat-driven cuts with whip transitions.",
        contrast=0.18, saturation=1.18, vibrance=0.35, warmth=0.03, exposure_target=0.46,
        fps=30, pace=(1.0, 1.8), transitions={"cut": 0.7, "smoothleft": 0.15, "hblur": 0.15}, slowmo=False,
        punch=0.14, flash=True, energetic=True,
    ),
    "moody": Look(
        "moody", "Moody", "Darker, desaturated, cool shadows and faded blacks. Great for cities, rain and mountains.",
        contrast=0.22, saturation=0.72, shadows=(-0.02, 0.0, 0.03), highlights=(0.01, 0.005, -0.01),
        lift=0.045, gain=0.95, teal_greens=0.4, exposure_target=0.34, fps=24, vignette=0.5, grain=5,
        pace=(2.4, 3.8), transitions={"cut": 0.6, "fade": 0.4}, punch=0.0,
    ),
    "film": Look(
        "film", "Vintage Film", "Warm faded film stock with grain and a soft glow.",
        contrast=0.12, saturation=0.85, warmth=0.06, shadows=(0.01, 0.01, 0.03), highlights=(0.03, 0.015, -0.02),
        lift=0.07, gain=0.96, exposure_target=0.44, fps=24, vignette=0.4, grain=9, glow=0.18,
        pace=(2.0, 3.0), transitions={"cut": 0.6, "fade": 0.4},
    ),
    "golden": Look(
        "golden", "Golden Hour", "Sun-kissed warm tones with dreamy glow. Beaches and sunsets.",
        contrast=0.14, saturation=1.05, vibrance=0.15, warmth=0.09, highlights=(0.04, 0.02, -0.03),
        lift=0.02, exposure_target=0.45, fps=24, vignette=0.25, glow=0.25,
        pace=(2.0, 3.2), transitions={"cut": 0.6, "fade": 0.4},
    ),
    "clean": Look(
        "clean", "Natural", "True-to-life colour with a gentle contrast lift.",
        contrast=0.08, saturation=1.04, vibrance=0.1, fps=30, pace=(1.8, 2.8), transitions={"cut": 0.85, "fade": 0.15},
        punch=0.08,
    ),
    "noir": Look(
        "noir", "Black & White", "High-contrast monochrome with film grain.",
        contrast=0.35, mono=True, lift=0.01, exposure_target=0.40, fps=24, vignette=0.45, grain=8,
        pace=(2.2, 3.4), transitions={"cut": 0.7, "fadeblack": 0.3},
    ),
}

LUMA = np.array([0.2126, 0.7152, 0.0722])


def _hue(x):
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(-1), x.min(-1)
    d = np.where(mx - mn == 0, 1e-6, mx - mn)
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return h, mx - mn


def grade(x, look: Look):
    x = np.clip(x, 0, 1)
    if look.warmth:
        x = x * np.array([1 + look.warmth, 1 + look.warmth * 0.25, 1 - look.warmth])
    luma = (x @ LUMA)[..., None]
    if look.teal_greens:
        h, chroma = _hue(x)
        w = np.exp(-(((h - 110) / 40) ** 2)) * look.teal_greens  # greens/yellow-greens
        w = (w * np.clip(chroma / 0.15, 0, 1))[..., None]
        teal = luma + (x - luma) * 0.8
        teal = teal * np.array([0.92, 1.0, 1.06])
        x = x * (1 - w) + teal * w
    chroma = (x.max(-1) - x.min(-1))[..., None]
    sat = look.saturation + look.vibrance * (1 - np.clip(chroma / 0.5, 0, 1))
    x = luma + (x - luma) * sat
    if look.mono:
        m = x @ np.array([0.30, 0.59, 0.11])
        x = np.repeat(m[..., None], 3, axis=-1)
    x = np.clip(x, 0, 1)
    s = x * x * (3 - 2 * x)
    x = (1 - look.contrast) * x + look.contrast * s
    luma = (x @ LUMA)[..., None]
    x = x + (1 - luma) ** 2 * np.array(look.shadows) + luma ** 2 * np.array(look.highlights)
    x = look.lift + np.clip(x, 0, 1) * (look.gain - look.lift)
    return np.clip(x, 0, 1)


def render_pixels(rgb, profile: str, look_key: str | None, ev: float = 0.0):
    out = to_display(rgb, profile, ev)
    if look_key:
        out = grade(out, LOOKS[look_key])
    return out


def lut_path(profile: str, look_key: str | None, ev: float) -> Path:
    ev = round(ev * 3) / 3
    look_sig = repr(asdict(LOOKS[look_key])) if look_key else "none"
    off = LUT_DIR / f"{profile}.cube"
    off_sig = str(off.stat().st_mtime) if off.exists() else ""
    h = hashlib.sha1(f"v3|{profile}|{look_sig}|{ev:.3f}|{off_sig}".encode()).hexdigest()[:12]
    path = LUT_CACHE / f"{profile}_{look_key or 'none'}_{h}.cube"
    with _lut_lock:
        if not path.exists():
            _write_lut(path, profile, look_key, ev)
    return path


_lut_lock = threading.Lock()


def _write_lut(path: Path, profile: str, look_key: str | None, ev: float):
    n = LUT_SIZE
    axis = np.linspace(0, 1, n)
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
    grid = np.stack([r, g, b], axis=-1).reshape(-1, 3)
    out = render_pixels(grid, profile, look_key, ev)
    lines = [f"TITLE \"videocut {profile} {look_key}\"", f"LUT_3D_SIZE {n}"]
    lines += [f"{a:.6f} {bb:.6f} {c:.6f}" for a, bb, c in out]
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.rename(path)


def auto_exposure(samples: list[np.ndarray], profile: str, target: float = 0.42, rec709_limit: float = 0.7) -> float:
    """Pick the exposure (stops) that brings the median display luma near the target."""
    if not samples:
        return 0.0
    pix = np.concatenate([s.reshape(-1, 3) for s in samples])
    if len(pix) > 20000:
        pix = pix[np.random.default_rng(0).choice(len(pix), 20000, replace=False)]
    limit = rec709_limit if profile == "rec709" and official_lut(profile) is None else 2.0
    best, best_err = 0.0, 9.0
    for ev in np.arange(-limit, limit + 1e-6, 1 / 3):
        med = float(np.median(to_display(pix, profile, ev) @ LUMA))
        err = abs(med - target)
        if err < best_err:
            best, best_err = float(ev), err
    base = to_display(pix, profile, 0.0) @ LUMA
    if np.median(base) < 0.16 and np.percentile(base, 97) > 0.45:
        best = min(best, 1 / 3)  # night scene with lights: keep it dark
    if profile == "rec709":
        best *= 0.6  # baked footage: only nudge, never fully normalise
    return round(best * 3) / 3


def looks_catalog():
    return [{"key": k, "name": v.name, "blurb": v.blurb, "fps": v.fps} for k, v in LOOKS.items()]
