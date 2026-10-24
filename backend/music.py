"""Music structure for editing: beats, downbeats/bars, energy, sections, drops, best excerpt, and cut slots."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import control
from .config import DATA_DIR, FFMPEG

CACHE = DATA_DIR / "music_cache"
CACHE.mkdir(parents=True, exist_ok=True)
VERSION = 2
SR = 22050
HOP = 512


def load_audio(path: Path, max_seconds: float = 480) -> np.ndarray:
    r = control.run([FFMPEG, "-v", "error", "-t", str(max_seconds), "-i", str(path), "-vn", "-ac", "1",
                     "-ar", str(SR), "-f", "f32le", "-"], check=True)
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def _z(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-9)


def _norm_db(db, min_range: float = 12.0):
    """Loudness (dB) to 0-1 against the song's own range, but never stretching less than min_range dB."""
    db = np.asarray(db, float)
    lo, hi = np.percentile(db, 3), np.percentile(db, 97)
    span = max(hi - lo, min_range)
    return np.clip((db - (hi - span)) / span, 0, 1)


def _cache_file(path: Path) -> Path:
    st = path.stat()
    key = hashlib.sha1(f"{VERSION}|{path.resolve()}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:16]
    return CACHE / f"{key}.json"


def analyze(path: Path) -> dict:
    path = Path(path)
    cf = _cache_file(path)
    if cf.exists():
        return json.loads(cf.read_text())
    info = analyze_signal(load_audio(path))
    info["name"] = path.name
    cf.write_text(json.dumps(info))
    return info


def fit_constant_grid(onset: np.ndarray, dp_beats: np.ndarray, duration: float) -> np.ndarray | None:
    """Fit a fixed-tempo beat grid to the onset envelope. The DP tracker's integer-frame period drifts."""
    fps = SR / HOP
    p0 = float(np.median(np.diff(dp_beats)))
    frames = np.arange(len(onset))
    env = np.maximum.reduce([np.roll(onset, k) for k in (-1, 0, 1)])
    env = env / (env.max() + 1e-9)
    lo, hi = float(dp_beats[0]) - p0 / 2, float(dp_beats[-1]) + p0 / 2

    def score(times):
        times = times[(times >= lo) & (times <= hi)]
        return float(np.interp(times * fps, frames, env).mean()) if len(times) else 0.0

    best = (-1.0, p0, 0.0)
    for period in np.linspace(p0 * 0.97, p0 * 1.03, 241):
        n = int((duration - 0.1) / period)
        k = np.arange(n)
        for phase in np.linspace(0, period, 48, endpoint=False):
            s = score(phase + k * period)
            if s > best[0]:
                best = (s, period, phase)
    s, period, phase = best
    fine = np.linspace(phase - period / 48, phase + period / 48, 21)
    k = np.arange(int((duration - 0.1) / period))
    phase = max(fine, key=lambda ph: score(ph + k * period))
    grid = phase + k * period
    if score(grid) < 0.9 * score(dp_beats):
        return None
    return grid[grid < duration - 0.05]


def analyze_signal(y: np.ndarray) -> dict:
    import librosa

    duration = len(y) / SR
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    mel_db = librosa.power_to_db(librosa.feature.melspectrogram(S=S ** 2, sr=SR, n_mels=64))
    onset = librosa.onset.onset_strength(S=mel_db, sr=SR)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=SR, hop_length=HOP, units="frames")
    beats = np.asarray(beats, int)
    tempo = float(np.atleast_1d(tempo)[0])
    if len(beats) < 16:
        bt = librosa.frames_to_time(beats, sr=SR, hop_length=HOP).tolist()
        return {"tempo": tempo, "duration": duration, "beats": bt, "downbeat_phase": 0, "bars": bt[::4],
                "beat_energy": [0.5] * len(bt), "bar_energy": [0.5] * len(bt[::4]), "sections": [], "drops": []}
    beat_t = librosa.frames_to_time(beats, sr=SR, hop_length=HOP)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    sub = librosa.power_to_db(S[(freqs > 30) & (freqs < 150)] ** 2)
    low = librosa.onset.onset_strength(S=sub, sr=SR)
    kick_weighted = 0.65 * low / (low.max() + 1e-9) + 0.35 * onset / (onset.max() + 1e-9)
    grid_t = fit_constant_grid(kick_weighted, beat_t, duration)
    if grid_t is not None:
        beat_t = grid_t
        beats = np.clip(np.round(beat_t * SR / HOP).astype(int), 0, S.shape[1] - 1)
        tempo = 60.0 / float(np.median(np.diff(beat_t)))
    chroma = librosa.feature.chroma_stft(S=S ** 2, sr=SR, hop_length=HOP)
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=13)
    rms = librosa.feature.rms(S=S, frame_length=2048, hop_length=HOP)[0]

    edges = np.concatenate([beats, [S.shape[1]]])

    def per_beat(feat):
        return np.array([feat[..., a:max(a + 1, b)].mean(axis=-1) for a, b in zip(edges[:-1], edges[1:])])

    low_b = np.array([low[max(0, a - 1):a + 3].max() for a in beats])
    chroma_b = per_beat(chroma)
    cn = chroma_b / (np.linalg.norm(chroma_b, axis=1, keepdims=True) + 1e-9)
    chroma_nov = np.r_[0.0, 1 - (cn[1:] * cn[:-1]).sum(1)]
    rms_db = 20 * np.log10(per_beat(rms) + 1e-6)
    onset_b = per_beat(onset)

    # Downbeat phase (4/4): the 1 carries the strongest kick and most harmonic changes.
    zl, zc = _z(low_b), _z(chroma_nov)
    phase = int(np.argmax([zl[p::4].mean() + 1.5 * zc[p::4].mean() for p in range(4)]))

    onset_db = 20 * np.log10(onset_b + 1e-3)
    beat_energy = 0.75 * _norm_db(rms_db) + 0.25 * _norm_db(onset_db)
    beat_energy = np.convolve(beat_energy, np.ones(3) / 3, mode="same")
    period = float(np.median(np.diff(beat_t)))
    beat_mfcc = per_beat(mfcc)
    nb_beats = len(beats)

    def structure(phase: int):
        down_idx = np.arange(phase, nb_beats, 4)
        nbars = len(down_idx)
        bar_slices = [slice(d, min(d + 4, nb_beats)) for d in down_idx]
        bar_db = np.array([rms_db[s].mean() for s in bar_slices])
        bar_energy = 0.75 * _norm_db(bar_db) + 0.25 * _norm_db(np.array([onset_db[s].mean() for s in bar_slices]))
        bar_t = beat_t[down_idx]
        timbre = np.array([beat_mfcc[s].mean(axis=0) for s in bar_slices])
        timbre = (timbre - timbre.mean(0)) / (timbre.std(0) + 1e-9)
        nov_t, nov_e = np.zeros(nbars), np.zeros(nbars)
        for b in range(1, nbars):
            nov_t[b] = np.linalg.norm(timbre[max(0, b - 2):b].mean(0) - timbre[b:b + 2].mean(0))
            nov_e[b] = abs(bar_energy[b:b + 2].mean() - bar_energy[max(0, b - 2):b].mean())
        nov = 0.4 * nov_t / (nov_t.max() + 1e-9) + nov_e / (nov_e.max() + 1e-9)
        thr = max(0.35, np.median(nov) + 0.75 * nov.std())
        peaks = [b for b in range(1, nbars - 1) if nov[b] >= nov[b - 1] and nov[b] >= nov[b + 1] and nov[b] > thr]
        boundaries: list[int] = []
        for b in sorted(peaks, key=lambda b: -nov[b]):
            if all(abs(b - o) >= 2 for o in boundaries):
                boundaries.append(b)
        boundaries.sort()
        cuts = [0] + boundaries + [nbars]
        sections = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            end_t = float(bar_t[b]) if b < nbars else min(duration, float(bar_t[-1] + 4 * period))
            sections.append({"start": float(bar_t[a]), "end": end_t, "bar0": int(a), "bar1": int(b),
                             "energy": round(float(bar_energy[a:b].mean()), 3)})
        drops = []
        for b in boundaries:
            before = bar_energy[max(0, b - 2):b].mean()
            after = bar_energy[b:b + 2].mean()
            jump_db = bar_db[b:b + 2].mean() - bar_db[max(0, b - 2):b].mean()
            if after - before > 0.25 and after > 0.55 and jump_db > 4.0:
                # land on the exact beat where the kick comes in, within a beat or two of the bar line
                d = int(down_idx[b])
                cand = range(max(1, d - 2), min(nb_beats - 1, d + 3))
                hit = max(cand, key=lambda i: zl[i] + 2 * (beat_energy[i] - beat_energy[i - 1]))
                drops.append({"t": float(beat_t[hit]), "beat": hit, "bar": int(b), "strength": round(float(after - before), 3)})
        return bar_t, bar_energy, sections, drops

    bar_t, bar_energy, sections, drops = structure(phase)
    if drops:
        drop_phases = [d["beat"] % 4 for d in drops]
        voted = max(set(drop_phases), key=drop_phases.count)
        if voted != phase:  # a drop is the most reliable downbeat there is
            phase = voted
            bar_t, bar_energy, sections, drops = structure(phase)

    return {
        "tempo": round(tempo, 2),
        "duration": duration,
        "period": period,
        "beats": [round(float(t), 4) for t in beat_t],
        "downbeat_phase": phase,
        "bars": [round(float(t), 4) for t in bar_t],
        "beat_energy": [round(float(e), 3) for e in beat_energy],
        "bar_energy": [round(float(e), 3) for e in bar_energy],
        "sections": sections,
        "drops": drops,
    }


def pick_excerpt(info: dict, target: float, mode: str = "auto") -> dict:
    """Choose where in the song the edit runs. Starts and ends on bar lines."""
    bars = list(info["bars"])
    period = info.get("period") or 0.5
    bars.append(bars[-1] + 4 * period) if bars else None
    nb = len(bars) - 1
    energy = info["bar_energy"]
    boundary_bars = {s["bar0"] for s in info["sections"]}
    drops = [d["t"] for d in info["drops"]]
    if nb < 2:
        start = info["beats"][0] if info["beats"] else 0.0
        return {"start": start, "end": min(info["duration"], start + target), "bar0": 0, "bar1": nb}

    def end_for(s):
        best, best_err = None, 9e9
        for e in range(s + 1, nb + 1):
            span = bars[e] - bars[s]
            err = abs(span - target) - (0.15 * target if (e - s) % 4 == 0 and abs(span - target) < 0.12 * target else 0)
            if err < best_err:
                best, best_err = e, err
            if span > target * 1.3:
                break
        return best

    starts = [0] if mode == "start" else [s for s in range(nb) if s % 4 == 0 or s in boundary_bars]
    best, best_score = None, -9e9
    for s in starts:
        e = end_for(s)
        if e is None or bars[e] > info["duration"] + 0.05:
            continue
        t0, t1 = bars[s], bars[e]
        score = 0.6 * float(np.mean(energy[s:e])) if e > s else 0
        score += 0.1 if (s in boundary_bars or s == 0) else 0
        score -= 0.25 * abs((t1 - t0) - target) / target
        rels = [(d - t0) / max(1e-6, t1 - t0) for d in drops]
        if any(0.12 <= r <= 0.5 for r in rels):
            score += 0.6  # build-up first, then the drop lands early: the classic reel shape
        elif any(0 <= r < 0.12 for r in rels):
            score += 0.25
        if mode == "start":
            score = 0
        if score > best_score:
            best, best_score = (s, e), score
    if best is None:
        return {"start": bars[0], "end": min(info["duration"], bars[0] + target), "bar0": 0, "bar1": nb}
    s, e = best
    return {"start": float(bars[s]), "end": float(bars[e]), "bar0": s, "bar1": e}


def plan_slots(info: dict, excerpt: dict, pace: tuple[float, float], energetic: bool) -> tuple[list[dict], dict]:
    """Cut points inside the excerpt. Busy music gets shorter shots; cuts always land on the drop and section starts."""
    start, end = excerpt["start"], excerpt["end"]
    period = info.get("period") or 0.5
    bt = np.asarray(info["beats"])
    be = np.asarray(info["beat_energy"])
    grid = [float(t) for t in bt if start - 1e-3 <= t < end - period * 0.4]
    if not grid or abs(grid[0] - start) > 1e-3:
        grid.insert(0, start)
    grid.append(end)
    grid = np.asarray(grid)
    e_grid = np.interp(grid, bt, be) if len(bt) else np.full(len(grid), 0.5)
    downs = set(np.round(np.asarray(info["bars"]), 3))
    is_down = np.array([round(float(t), 3) in downs for t in grid])
    drops = [d["t"] for d in info["drops"] if start - 1e-3 <= d["t"] < end - period]
    sec_starts = [s["start"] for s in info["sections"] if start + period < s["start"] < end - period]

    def nearest(t):
        return int(np.argmin(np.abs(grid - t)))

    drop_idx = {nearest(d) for d in drops}
    forced = sorted(drop_idx | {nearest(t) for t in sec_starts})
    lo, hi = pace
    last = len(grid) - 1
    slots, p = [], 0
    while p < last:
        e = float(e_grid[p:min(last, p + 4)].mean())
        want = (hi * 1.25) + (lo * 0.75 - hi * 1.25) * e
        steps = max(1, int(round(want / period)))
        allowed = [1, 2, 4, 8] if (energetic and e > 0.7) else [2, 4, 8]
        steps = min(allowed, key=lambda a: (abs(a - steps), -a))
        to_drop = [d for d in drop_idx if d > p]
        if energetic and to_drop and (min(to_drop) - p) <= 8:
            steps = 1 if (min(to_drop) - p) <= 4 else 2
        nxt = min(last, p + steps)
        if steps >= 4:
            downs_ahead = [i for i in range(p + 2, min(last, p + steps + 2) + 1) if is_down[i] or i == last]
            if downs_ahead:
                nxt = min(downs_ahead, key=lambda i: abs(i - (p + steps)))
        f = [i for i in forced if p < i < nxt]
        if f:
            nxt = f[0]
        slots.append({"t0": float(grid[p] - start), "t1": float(grid[nxt] - start), "energy": round(e, 3),
                      "drop": p in drop_idx, "section_start": p in forced and p not in drop_idx,
                      "pre_drop": nxt in drop_idx})
        p = nxt
    while len(slots) > 1 and slots[-1]["t1"] - slots[-1]["t0"] < max(0.6, lo * 0.5):
        tail = slots.pop()
        slots[-1]["t1"] = tail["t1"]
    mask = (bt >= start) & (bt <= end)
    music_map = {
        "start": start, "end": end, "tempo": info["tempo"],
        "beats": [round(float(t - start), 3) for t in bt[mask]],
        "energy": [round(float(x), 3) for x in be[mask]],
        "bars": [round(float(t - start), 3) for t in info["bars"] if start - 1e-3 <= t <= end],
        "drops": [round(d - start, 3) for d in drops],
        "sections": [round(t - start, 3) for t in sec_starts],
    }
    return slots, music_map
