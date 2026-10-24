import numpy as np
import pytest

from backend import director, grouping, looks
from backend.looks import LOOKS, to_display


def clip(cid, when, dur=10, gps=None, camera="Sony", fps=25):
    return {"id": cid, "captured": when, "duration": dur, "gps": gps, "camera": camera, "orientation": "landscape",
            "is_drone": False, "fps": fps, "width": 1920, "height": 1080, "name": cid, "path": cid, "profile": "rec709"}


def test_grouping_splits_on_gap_distance_and_night():
    clips = [
        clip("a", "2026-09-12T09:00:00", gps=(-8.80, 115.10)),
        clip("b", "2026-09-12T09:20:00", gps=(-8.80, 115.10)),
        clip("c", "2026-09-12T10:00:00", gps=(-8.40, 115.30)),  # ~50 km away
        clip("d", "2026-09-12T18:00:00"),  # long gap
        clip("e", "2026-09-13T08:00:00"),  # next day
    ]
    groups = grouping.group_by_moment(clips, gap_hours=3, km=10)
    assert [g["clip_ids"] for g in groups] == [["a", "b"], ["c"], ["d"], ["e"]]
    assert groups[3]["days"] == [2]


@pytest.mark.parametrize("profile,grey_code", [("slog3", 420), ("flog", 470), ("flog2", 400)])
def test_log_mid_grey_maps_to_mid_display(profile, grey_code):
    rgb = np.full((1, 3), (grey_code - 64) / 876)
    out = to_display(rgb, profile)
    assert 0.36 < out[0, 0] < 0.46


def test_rec709_passthrough_and_grade_bounds():
    rgb = np.random.default_rng(0).random((500, 3))
    assert np.allclose(to_display(rgb, "rec709"), rgb)
    for look in LOOKS.values():
        g = looks.grade(rgb, look)
        assert g.min() >= 0 and g.max() <= 1


def test_mono_look_has_no_colour():
    g = looks.grade(np.random.default_rng(1).random((100, 3)), LOOKS["noir"])
    assert np.allclose(g[:, 0], g[:, 1]) and np.allclose(g[:, 1], g[:, 2])


def test_gamut_matrix_preserves_white():
    for src in ("sgamut3cine", "rec2020", "dgamut"):
        assert np.allclose(looks.gamut_matrix(src) @ np.ones(3), np.ones(3), atol=1e-6)


def fake_analysis(n=30, fps=3.0, seed=0):
    rng = np.random.default_rng(seed)
    return {"fps": fps, "ev": 0.0, "t": [i / fps + 1 / (2 * fps) for i in range(n)],
            "sharp": list(rng.random(n) * 100), "bright": [0.4] * n, "clip_hi": [0.0] * n, "clip_lo": [0.0] * n,
            "color": list(rng.random(n)), "motion": list(rng.random(n) * 0.05), "dx": [0.0] * n, "dy": [0.0] * n,
            "faces": [0] * n, "face_x": [None] * n, "face_y": [None] * n, "audio": [0.0] * n,
            "colx": [list(np.full(24, 1 / 24))] * n, "coly": [list(np.full(24, 1 / 24))] * n,
            "sig": [list(rng.random(12))] * n}


def test_plan_hits_target_and_respects_clip_bounds():
    clips = [clip(f"c{i}", f"2026-09-12T09:{i:02d}:00", dur=10, fps=60 if i % 2 else 25) for i in range(6)]
    ans = {c["id"]: fake_analysis(seed=i) for i, c in enumerate(clips)}
    plan = director.plan_edit(clips, ans, {"look": "cinematic", "platform": "vertical", "duration": 20})
    assert abs(plan["duration"] - 20) < 0.01
    for s in plan["shots"]:
        assert s["start"] >= 0 and s["start"] + s["src_dur"] <= 10 + 1e-6
        assert s["crop"]["mode"] == "crop"
    assert plan["shots"][0]["transition_in"]["type"] == "cut"
    assert any(s["speed"] < 1 for s in plan["shots"])  # 60p clips get slow-mo in the cinematic look


def test_music_slots_drive_the_plan():
    clips = [clip(f"c{i}", f"2026-09-12T09:{i:02d}:00", dur=12, fps=60) for i in range(6)]
    ans = {c["id"]: fake_analysis(n=36, seed=i) for i, c in enumerate(clips)}
    slots = [{"t0": 0.0, "t1": 2.0, "energy": 0.3, "drop": False, "section_start": False, "pre_drop": False},
             {"t0": 2.0, "t1": 4.0, "energy": 0.4, "drop": False, "section_start": False, "pre_drop": True},
             {"t0": 4.0, "t1": 5.0, "energy": 0.9, "drop": True, "section_start": False, "pre_drop": False},
             {"t0": 5.0, "t1": 6.0, "energy": 0.9, "drop": False, "section_start": False, "pre_drop": False},
             {"t0": 6.0, "t1": 8.0, "energy": 0.9, "drop": False, "section_start": True, "pre_drop": False}]
    plan = director.plan_edit(clips, ans, {"look": "vibrant", "platform": "youtube", "duration": 8}, slots)
    shots = plan["shots"]
    assert [s["timeline"] for s in shots] == [[s["t0"], s["t1"]] for s in slots]
    assert shots[2]["transition_in"]["type"] == "cut" and shots[2]["flash"] and shots[2]["punch"] > 0
    assert shots[1]["ramp"][1] < shots[1]["ramp"][0]  # decelerate into the drop
    for s in shots:
        assert s["start"] + s["src_dur"] <= 12 + 1e-6


@pytest.fixture(scope="module")
def song_info():
    from backend import music
    from scripts.make_samples import synth_song

    y, sr, drops = synth_song(bpm=120)
    assert sr == music.SR
    return music.analyze_signal(y), drops


def test_music_tempo_bars_and_drops(song_info):
    info, drops = song_info
    assert abs(info["tempo"] - 120) < 0.5
    assert abs(((info["bars"][0] / 2.0) % 1) - 0) < 0.05 or abs(((info["bars"][0] / 2.0) % 1) - 1) < 0.05
    found = [d["t"] for d in info["drops"]]
    assert len(found) == len(drops)
    assert all(abs(f - d) < 0.1 for f, d in zip(found, drops))


def test_excerpt_builds_into_the_drop_and_slots_cut_on_it(song_info):
    from backend import music

    info, drops = song_info
    ex = music.pick_excerpt(info, 15)
    assert abs((ex["end"] - ex["start"]) - 15) < 2.5
    inside = [d for d in drops if ex["start"] < d < ex["end"]]
    assert inside and (inside[0] - ex["start"]) > 1.5  # some build-up before the drop
    slots, mm = music.plan_slots(info, ex, (1.0, 1.8), True)
    drop_slots = [s for s in slots if s["drop"]]
    assert drop_slots and abs(drop_slots[0]["t0"] - (inside[0] - ex["start"])) < 0.1
    assert abs(slots[-1]["t1"] - (ex["end"] - ex["start"])) < 1e-6
    calm = [s["t1"] - s["t0"] for s in slots if s["energy"] < 0.4]
    busy = [s["t1"] - s["t0"] for s in slots if s["energy"] > 0.7 and not s["pre_drop"]]
    if calm and busy:
        assert np.mean(calm) > np.mean(busy)


def test_rotate_fix_swaps_dimensions_and_builds_filters():
    from backend import media

    c = {"width": 2160, "height": 3840, "orientation": "portrait"}
    media.set_rotate_fix(c, 90)
    assert (c["width"], c["height"], c["orientation"], c["rotate_fix"]) == (3840, 2160, "landscape", 90)
    media.set_rotate_fix(c, 180)  # from 90 to 180 is another quarter turn
    assert (c["width"], c["height"], c["orientation"]) == (2160, 3840, "portrait")
    media.set_rotate_fix(c, 360)
    assert c["rotate_fix"] == 0 and c["orientation"] == "portrait"
    assert media.rotate_filter(90) == "transpose=clock," and media.rotate_filter(-90) == "transpose=cclock,"
    assert media.rotate_filter(0) == ""


def test_flat_music_has_no_drops():
    from backend import music

    sr = music.SR
    t = np.arange(int(40 * sr)) / sr
    beat = 0.5
    y = (np.sin(2 * np.pi * 60 * t) * np.exp(-12 * (t % beat)) + 0.2 * np.sin(2 * np.pi * 330 * t)).astype(np.float32) * 0.5
    assert music.analyze_signal(y)["drops"] == []
