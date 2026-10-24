"""VideoCut API: scan a trip folder, group it, auto-edit reels, preview and export."""
from __future__ import annotations

import re
import shutil
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import ai, analysis, caption, control, director, grouping, media, render
from . import music as music_mod
from .config import EXPORT_DIR, MUSIC_DIR, ROOT
from .looks import LOOKS, PROFILES, looks_catalog
from .store import Project, get_project, jobs, list_projects, open_project

app = FastAPI(title="VideoCut")


@app.middleware("http")
async def revalidate_frontend(request, call_next):
    resp = await call_next(request)
    if not request.url.path.startswith("/media/"):
        resp.headers["Cache-Control"] = "no-cache"  # browsers kept showing an old UI after updates
    return resp
MUSIC_EXTS = {".mp3", ".m4a", ".aac", ".wav", ".flac", ".aiff", ".ogg"}


def need_project(pid: str) -> Project:
    p = get_project(pid)
    if not p:
        raise HTTPException(404, "Project not found")
    return p


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")[:60] or "reel"


# ---------------------------------------------------------------- pipeline steps
def thumb_paths(p: Project, cid: str):
    return p.thumbs / f"{cid}.jpg", p.thumbs / f"{cid}_sprite.jpg", p.thumbs / f"{cid}_sheet.jpg"


def make_thumbs(p: Project, clip: dict, force: bool = False):
    thumb, sprite, sheet = thumb_paths(p, clip["id"])
    if force or not thumb.exists():
        render.clip_frames(clip, sprite, thumb, sheet)


def regroup(p: Project, keep_merged: bool = True):
    """Group by day / gap / place, keeping groups the user merged and names they gave."""
    s = p.data.get("settings", {})
    clips = [c for c in p.clips.values() if not c.get("excluded")]
    old = p.data.get("groups", [])
    titles = {g["id"]: g["title"] for g in old if g.get("renamed")}
    kept = []
    if keep_merged:
        first = grouping.trip_first_day(clips) if clips else None
        live = {c["id"]: c for c in clips}
        for g in old:
            members = [live[i] for i in g["clip_ids"] if i in live]
            if g.get("kind") == "merged" and members:
                ng = grouping.summarize(members, first, title=g["title"], kind="merged")
                ng.update(id=g["id"], cover=g.get("cover") if g.get("cover") in live else members[0]["id"],
                          renamed=g.get("renamed", False))
                kept.append(ng)
    taken = {i for g in kept for i in g["clip_ids"]}
    groups = grouping.group_by_moment([c for c in clips if c["id"] not in taken], float(s.get("gap_hours", 3)),
                                      float(s.get("km", 10)))
    for g in groups:
        g["cover"] = g["clip_ids"][len(g["clip_ids"]) // 2]
        if g["id"] in titles:
            g.update(title=titles[g["id"]], renamed=True)
    groups = sorted(groups + kept, key=lambda g: g["start"])
    p.update(lambda d: d.__setitem__("groups", groups))


def scan(p: Project, progress):
    folder = Path(p.data["folder"])
    progress(0.02, "Looking for videos")
    vids = media.find_videos(folder)
    if not vids:
        raise RuntimeError(f"No video files found in {folder}")
    progress(0.05, f"Reading metadata of {len(vids)} videos")
    exif = media.exif_batch([v.resolve() for v in vids])
    old = {c["path"]: c for c in p.clips.values()}
    clips: dict[str, dict] = {}
    done = 0

    def probe(v: Path):
        c = media.probe_clip(v, exif.get(str(v.resolve()), {}))
        if c and (prev := old.get(c["path"])) and prev["id"] == c["id"]:
            for keep in ("profile_override", "excluded", "tags", "starred", "smart"):
                if keep in prev:
                    c[keep] = prev[keep]
            if prev.get("profile_override"):
                c["profile"] = prev["profile_override"]
            if prev.get("rotate_fix"):
                media.set_rotate_fix(c, prev["rotate_fix"])
        return c

    with ThreadPoolExecutor(4) as ex:
        for c in ex.map(control.bind(probe), vids):
            done += 1
            if c:
                clips[c["id"]] = c
            progress(0.05 + 0.35 * done / len(vids), f"Probed {done}/{len(vids)}")
    p.update(lambda d: d.__setitem__("clips", clips))
    done = 0
    try:
        with ThreadPoolExecutor(4) as ex:
            for _ in ex.map(control.bind(lambda c: make_thumbs(p, c)), clips.values()):
                done += 1
                progress(0.4 + 0.55 * done / max(1, len(clips)), f"Thumbnails {done}/{len(clips)}")
    finally:
        regroup(p)  # even if cancelled: groups work without every thumbnail
    p.update(lambda d: d.__setitem__("scanned", datetime.now().isoformat(timespec="seconds")))
    return {"clips": len(clips), "groups": len(p.data["groups"])}


def analyze_clips(clips: list[dict], progress, lo: float, hi: float) -> dict[str, dict]:
    out, done = {}, 0
    with ThreadPoolExecutor(3) as ex:
        for c, an in zip(clips, ex.map(control.bind(analysis.analyze_clip), clips)):
            out[c["id"]] = an
            done += 1
            progress(lo + (hi - lo) * done / len(clips), f"Analysed {done}/{len(clips)} clips")
    return out


def make_edit(p: Project, gid: str, settings: dict, progress):
    g = p.group(gid)
    if not g:
        raise RuntimeError("Group not found")
    clips = [p.clips[i] for i in g["clip_ids"] if i in p.clips and not p.clips[i].get("excluded")]
    progress(0.01, f"Analysing {len(clips)} clips")
    analyses = analyze_clips(clips, progress, 0.02, 0.45)
    music_info, slots = None, None
    if settings.get("music"):
        mpath = MUSIC_DIR / Path(settings["music"]).name
        if mpath.exists():
            progress(0.46, "Listening to the music: beats, bars, drops")
            info = music_mod.analyze(mpath)
            look = LOOKS[settings["look"]]
            excerpt = music_mod.pick_excerpt(info, float(settings["duration"]), settings.get("music_section", "auto"))
            slots, music_map = music_mod.plan_slots(info, excerpt, look.pace, look.energetic)
            music_info = {"path": str(mpath), "offset": excerpt["start"], "tempo": round(info["tempo"]), "map": music_map}
    notes: list[str] = []
    if settings.get("smart_picks"):
        notes += smart_picks(p, clips, analyses, progress)
        clips = [p.clips[c["id"]] for c in clips]
    progress(0.48, "Choosing the best moments")
    mm = music_info["map"] if music_info else {}
    plan = director.plan_edit(clips, analyses, settings, slots, beats=mm.get("beats"), bars=mm.get("bars"),
                              refine=analysis.refine_action)
    plan["text"] = {"place": (settings.get("place") or "").strip(), "show_date": bool(settings.get("show_date")), "notes": {}}
    eid = uuid.uuid4().hex[:8]
    out_dir = p.renders / eid
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        res = render.render_plan(plan, {c["id"]: c for c in clips}, out_dir / "preview.mp4", out_dir / "work", "preview",
                                 music_info, lambda f, m: progress(0.5 + 0.5 * f, m))
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    rec = {"id": eid, "group_id": gid, "created": datetime.now().isoformat(timespec="seconds"), "settings": settings,
           "plan": plan, "music": music_info, "notes": notes,
           "preview": res["file"], "thumbs": res["thumbs"], "final": None}

    def save(d):
        lst = d.setdefault("edits", {}).setdefault(gid, [])
        lst.insert(0, rec)
        for stale in lst[12:]:
            shutil.rmtree(p.renders / stale["id"], ignore_errors=True)
        del lst[12:]

    p.update(save)
    return {"edit_id": eid, "group_id": gid}


def smart_picks(p: Project, clips: list[dict], analyses: dict[str, dict], progress) -> list[str]:
    """Have Claude rate each clip's best moments once; failures only skip the feature."""
    todo = [c for c in clips if not (c.get("smart") or {}).get("version")]
    if not todo:
        return []
    if not ai.available():
        return ["Smart picks skipped: no Claude API key is set."]
    sc = director.score_samples(clips, analyses)
    strips = {}
    for c in todo:
        an, s = analyses[c["id"]], sc[c["id"]]["score"]
        order = [int(i) for i in np.argsort(-s)]
        times: list[float] = []
        for i in order:  # three good moments, at least 1.5 s apart
            t = an["t"][i]
            if all(abs(t - u) > 1.5 for u in times):
                times.append(t)
            if len(times) == 3:
                break
        strips[c["id"]] = render.moment_strip(c, sorted(times), p.thumbs / f"{c['id']}_moments.jpg")
    progress(0.47, f"Claude is rating {len(todo)} clips")
    try:
        ratings = ai.rate_moments(todo, lambda c: strips.get(c["id"]))
    except Exception as e:
        return [f"Smart picks skipped: {str(e)[:160]}"]

    def save(d):
        for cid, r in ratings.items():
            if cid in d["clips"]:
                d["clips"][cid]["smart"] = r
    p.update(save)
    return []


def music_for(e: dict) -> dict | None:
    """An edit's soundtrack, found by file name so edits survive the project folder moving."""
    m = e.get("music")
    return dict(m, path=str(MUSIC_DIR / Path(m["path"]).name)) if m else None


def rerender_preview(p: Project, eid: str, progress):
    """Re-render an edit's preview with its existing plan (used after changing on-screen text)."""
    e = find_edit(p, eid)
    out_dir = p.renders / eid
    name = f"preview-{datetime.now().strftime('%H%M%S')}.mp4"
    clips = {c["id"]: c for c in p.clips.values()}
    res = render.render_plan(e["plan"], clips, out_dir / name, out_dir / "work", "preview", music_for(e), progress)
    old = [e["preview"], *e.get("thumbs", [])]

    def save(d):
        e.update(preview=res["file"], thumbs=res["thumbs"], final=None)
    p.update(save)
    for f in old:
        (out_dir / f).unlink(missing_ok=True)
    return {"edit_id": eid}


def find_edit(p: Project, eid: str) -> dict:
    for lst in p.data.get("edits", {}).values():
        for e in lst:
            if e["id"] == eid:
                return e
    raise HTTPException(404, "Edit not found")


def export_final(p: Project, eid: str, progress):
    e = find_edit(p, eid)
    plan = e["plan"]
    g = p.group(e["group_id"]) or {"title": "reel"}
    dest_dir = EXPORT_DIR / slug(p.data["name"])
    dest_dir.mkdir(parents=True, exist_ok=True)
    base = f"{slug(g['title'])}_{plan['look']}_{plan['platform']}_{round(plan['duration'])}s"
    dest = dest_dir / f"{base}.mp4"
    n = 2
    while dest.exists():
        dest = dest_dir / f"{base}-{n}.mp4"
        n += 1
    clips = {c["id"]: c for c in p.clips.values()}
    try:
        music = music_for(e)
        render.render_plan(plan, clips, dest, p.renders / eid / "final_work", "final", music, progress)
    except BaseException:
        dest.unlink(missing_ok=True)
        shutil.rmtree(p.renders / eid / "final_work", ignore_errors=True)
        raise
    p.update(lambda d: e.__setitem__("final", str(dest)))
    return {"path": str(dest)}


def ai_themes(p: Project, progress):
    clips = [c for c in p.clips.values() if not c.get("excluded")]
    todo = [c for c in clips if not c.get("tags")]
    if todo:
        tags = ai.tag_clips(todo, lambda c: thumb_paths(p, c["id"])[2], lambda f, m: progress(0.05 + 0.75 * f, m))

        def put(d):
            for cid, t in tags.items():
                if cid in d["clips"]:
                    d["clips"][cid]["tags"] = t
        p.update(put)
    progress(0.85, "Grouping by theme")
    clips = [c for c in p.clips.values() if not c.get("excluded")]
    raw = ai.theme_groups(clips, {c["id"]: c.get("tags", {}) for c in clips})
    first = grouping.trip_first_day(clips)
    groups = []
    for g in raw:
        members = [p.clips[i] for i in g["clip_ids"]]
        sg = grouping.summarize(members, first, title=g["title"], kind="theme",
                                extra={"description": g["description"], "suggested_look": g["suggested_look"],
                                       "music_mood": g["music_mood"]})
        best = max(members, key=lambda c: c.get("tags", {}).get("highlight", 0))
        sg["cover"] = best["id"]
        groups.append(sg)
    p.update(lambda d: d.update({"ai_groups": groups, "group_mode": "ai"}))
    return {"groups": len(groups)}


# ---------------------------------------------------------------- API
class OpenReq(BaseModel):
    folder: str


class GroupSettings(BaseModel):
    gap_hours: float = 3
    km: float = 10


class EditReq(BaseModel):
    duration: float = 20
    platform: str = "reels"
    look: str = "cinematic"
    music: str | None = None
    title: str = ""
    subtitle: str = ""
    seed: int = 0
    slowmo: bool = True
    hook: bool = True
    letterbox: bool = True
    music_section: str = "auto"
    ramps: bool = True
    ambient: float | None = None
    ev_bias: float = 0.0
    exclude_clips: list[str] = []
    place: str = ""
    show_date: bool = False
    smart_picks: bool = False


def ui_version() -> str:
    m = re.search(r"app\.js\?v=(\w+)", (ROOT / "frontend" / "index.html").read_text())
    return m.group(1) if m else "dev"


@app.get("/api/version")
def version():
    return {"ui": ui_version()}


@app.get("/api/meta")
def meta():
    return {"ui_version": ui_version(), "looks": looks_catalog(), "platforms": director.PLATFORMS, "ai": ai.available(),
            "profiles": {k: v[2] for k, v in PROFILES.items()}, "export_dir": str(EXPORT_DIR), "music_dir": str(MUSIC_DIR)}


@app.get("/api/projects")
def projects():
    return list_projects()


PICK_SCRIPT = [
    # Run the dialog inside System Events so it comes to the front instead of opening behind other windows.
    "-e", 'tell application "System Events"',
    "-e", "activate",
    "-e", 'set f to choose folder with prompt "Choose your footage folder"',
    "-e", "end tell",
    "-e", "POSIX path of f",
]


@app.post("/api/pick-folder")
def pick_folder():
    r = subprocess.run(["osascript", *PICK_SCRIPT], capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        return {"folder": r.stdout.strip().rstrip("/")}
    err = r.stderr.strip()
    if "-128" in err:  # the user pressed Cancel
        return {"folder": None, "cancelled": True}
    print(f"folder picker failed (exit {r.returncode}): {err}", flush=True)
    if "-1743" in err or "not allowed" in err.lower():
        msg = "macOS blocked the folder picker. Allow it in System Settings → Privacy & Security → Automation, or paste the folder path."
    else:
        msg = f"Couldn't open the folder picker ({err or 'no response'}). Paste the folder path instead."
    return {"folder": None, "error": msg}


@app.post("/api/projects/open")
def open_folder(req: OpenReq):
    folder = Path(req.folder).expanduser()
    if not folder.is_dir():
        raise HTTPException(400, f"Not a folder: {folder}")
    p = open_project(folder)
    job = None
    if not p.clips:
        job = jobs.start("scan", lambda pr: scan(p, pr), project_id=p.id)
    return {"project_id": p.id, "job": job}


@app.post("/api/projects/{pid}/rescan")
def rescan(pid: str):
    p = need_project(pid)
    return jobs.start("scan", lambda pr: scan(p, pr), project_id=pid)


@app.get("/api/projects/{pid}")
def project(pid: str):
    p = need_project(pid)
    d = p.data
    return {"id": p.id, "name": d["name"], "folder": d["folder"], "settings": d.get("settings", {}),
            "group_mode": d.get("group_mode", "moment"), "has_ai_groups": bool(d.get("ai_groups")),
            "clips": d.get("clips", {}), "groups": p.groups(), "edits": d.get("edits", {}),
            "jobs": [j for j in jobs.active() if j.get("project_id") == pid]}


@app.post("/api/projects/{pid}/regroup")
def do_regroup(pid: str, s: GroupSettings):
    p = need_project(pid)
    p.update(lambda d: d.update({"settings": {**d.get("settings", {}), **s.model_dump()}, "group_mode": "moment"}))
    regroup(p, keep_merged=False)  # the user asked for a fresh split; names of unchanged groups survive
    return {"groups": len(p.data["groups"])}


@app.post("/api/projects/{pid}/group-mode/{mode}")
def group_mode(pid: str, mode: str):
    p = need_project(pid)
    if mode not in ("moment", "ai"):
        raise HTTPException(400, "mode must be moment or ai")
    p.update(lambda d: d.__setitem__("group_mode", mode))
    return {"ok": True}


@app.post("/api/projects/{pid}/ai-themes")
def start_ai(pid: str):
    p = need_project(pid)
    if not ai.available():
        raise HTTPException(400, "Set ANTHROPIC_API_KEY to use Claude scene tagging")
    return jobs.start("ai", lambda pr: ai_themes(p, pr), project_id=pid)


class GroupPatch(BaseModel):
    title: str


@app.patch("/api/projects/{pid}/groups/{gid}")
def rename_group(pid: str, gid: str, body: GroupPatch):
    p = need_project(pid)

    def ren(d):
        for g in d.get("groups", []) + d.get("ai_groups", []):
            if g["id"] == gid:
                g["title"] = body.title.strip()[:80] or g["title"]
                g["renamed"] = True
    p.update(ren)
    return {"ok": True}


class MergeReq(BaseModel):
    group_ids: list[str]


@app.post("/api/projects/{pid}/groups/merge")
def merge_groups(pid: str, body: MergeReq):
    p = need_project(pid)
    key = "ai_groups" if p.data.get("group_mode") == "ai" and p.data.get("ai_groups") else "groups"
    groups = p.data.get(key, [])
    picked = [g for g in groups if g["id"] in body.group_ids]
    if len(picked) < 2:
        raise HTTPException(400, "Pick at least two groups")
    ids = [i for g in picked for i in g["clip_ids"]]
    clips = [p.clips[i] for i in ids if i in p.clips]
    parts = list(dict.fromkeys(g["title"].split(" · ")[0] for g in picked))
    merged = grouping.summarize(clips, grouping.trip_first_day(list(p.clips.values())),
                                title=" + ".join(parts)[:80], kind="merged")
    merged["cover"] = picked[0].get("cover", ids[0])
    rest = [g for g in groups if g["id"] not in body.group_ids]
    p.update(lambda d: d.__setitem__(key, sorted(rest + [merged], key=lambda g: g["start"])))
    return merged


class ClipPatch(BaseModel):
    profile: str | None = None
    excluded: bool | None = None
    rotate: int | None = None  # degrees clockwise to add: 90, 180 or -90
    starred: bool | None = None


@app.patch("/api/projects/{pid}/clips/{cid}")
def patch_clip(pid: str, cid: str, body: ClipPatch):
    p = need_project(pid)
    if cid not in p.clips:
        raise HTTPException(404, "Clip not found")
    if body.profile and body.profile not in PROFILES:
        raise HTTPException(400, "Unknown profile")

    def upd(d):
        c = d["clips"][cid]
        if body.profile:
            c["profile"] = c["profile_override"] = body.profile
        if body.excluded is not None:
            c["excluded"] = body.excluded
        if body.starred is not None:
            c["starred"] = body.starred
        if body.rotate:
            media.set_rotate_fix(c, c.get("rotate_fix", 0) + body.rotate)
    if body.rotate is not None and body.rotate % 90:
        raise HTTPException(400, "rotate must be a multiple of 90")
    p.update(upd)
    if body.profile or body.rotate:
        make_thumbs(p, p.clips[cid], force=True)
    return p.clips[cid]


@app.post("/api/projects/{pid}/groups/{gid}/edit")
def start_edit(pid: str, gid: str, req: EditReq):
    p = need_project(pid)
    if req.look not in LOOKS or req.platform not in director.PLATFORMS:
        raise HTTPException(400, "Unknown look or platform")
    if not p.group(gid):
        raise HTTPException(404, "Group not found")
    settings = req.model_dump()
    if settings["ambient"] is None:
        settings.pop("ambient")
    settings["duration"] = max(5.0, min(180.0, settings["duration"]))
    return jobs.start("edit", lambda pr: make_edit(p, gid, settings, pr), project_id=pid, group_id=gid)


@app.post("/api/projects/{pid}/edit-all")
def edit_all(pid: str, req: EditReq):
    p = need_project(pid)
    started = []
    for g in p.groups():
        settings = req.model_dump()
        if settings["ambient"] is None:
            settings.pop("ambient")
        if req.look == "auto":
            settings["look"] = g.get("suggested_look") or "cinematic"
        elif req.look not in LOOKS:
            raise HTTPException(400, "Unknown look")
        started.append(jobs.start("edit", lambda pr, gid=g["id"], s=settings: make_edit(p, gid, s, pr),
                                  project_id=pid, group_id=g["id"]))
    return started


@app.post("/api/projects/{pid}/edits/{eid}/final")
def start_final(pid: str, eid: str):
    p = need_project(pid)
    find_edit(p, eid)
    return jobs.start("final", lambda pr: export_final(p, eid, pr), project_id=pid, edit_id=eid)


class TextReq(BaseModel):
    place: str = ""
    show_date: bool = False
    notes: dict[str, str] = {}


@app.post("/api/projects/{pid}/edits/{eid}/text")
def set_text(pid: str, eid: str, body: TextReq):
    p = need_project(pid)
    e = find_edit(p, eid)
    n = len(e["plan"]["shots"])
    notes = {k: v.strip()[:120] for k, v in body.notes.items() if k.isdigit() and int(k) < n and v.strip()}

    def save(d):
        e["plan"]["text"] = {"place": body.place.strip()[:40], "show_date": body.show_date, "notes": notes}
        e["settings"].update(place=body.place.strip()[:40], show_date=body.show_date)
    p.update(save)
    return jobs.start("text", lambda pr: rerender_preview(p, eid, pr), project_id=pid, group_id=e["group_id"], edit_id=eid)


class CaptionReq(BaseModel):
    use_ai: bool = True


@app.post("/api/projects/{pid}/edits/{eid}/caption")
def post_caption(pid: str, eid: str, body: CaptionReq):
    p = need_project(pid)
    e = find_edit(p, eid)
    group = p.group(e["group_id"]) or {}
    out = caption.write(e["plan"], group, p.clips, body.use_ai)
    p.update(lambda d: e.__setitem__("post_caption", out))
    return out


@app.delete("/api/projects/{pid}/edits/{eid}")
def delete_edit(pid: str, eid: str):
    p = need_project(pid)

    def rm(d):
        for gid, lst in d.get("edits", {}).items():
            d["edits"][gid] = [e for e in lst if e["id"] != eid]
    p.update(rm)
    shutil.rmtree(p.renders / eid, ignore_errors=True)
    return {"ok": True}


@app.get("/api/jobs/{jid}")
def job(jid: str):
    j = jobs.get(jid)
    if not j:
        raise HTTPException(404, "No such job")
    return j


@app.post("/api/jobs/{jid}/{action}")
def job_action(jid: str, action: str):
    if not jobs.get(jid):
        raise HTTPException(404, "No such job")
    if action not in ("pause", "resume", "cancel"):
        raise HTTPException(400, "action must be pause, resume or cancel")
    return getattr(jobs, action)(jid)


@app.get("/api/music")
def music():
    return [{"name": f.name, "size": f.stat().st_size} for f in sorted(MUSIC_DIR.iterdir())
            if f.suffix.lower() in MUSIC_EXTS]


@app.post("/api/music")
async def upload_music(file: UploadFile):
    name = Path(file.filename or "track").name
    if Path(name).suffix.lower() not in MUSIC_EXTS:
        raise HTTPException(400, "Unsupported audio type")
    dest = MUSIC_DIR / name
    with dest.open("wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    return {"name": name}


class RevealReq(BaseModel):
    path: str


@app.post("/api/reveal")
def reveal(req: RevealReq):
    path = Path(req.path).resolve()
    if not path.exists() or EXPORT_DIR.resolve() not in path.parents:
        raise HTTPException(400, "Can only reveal exported videos")
    subprocess.run(["open", "-R", str(path)])
    return {"ok": True}


@app.get("/media/{pid}/thumbs/{name}")
def thumb_file(pid: str, name: str):
    p = need_project(pid)
    f = p.thumbs / Path(name).name
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f, headers={"Cache-Control": "max-age=3600"})


@app.get("/media/{pid}/look/{cid}/{look}.jpg")
def look_preview(pid: str, cid: str, look: str):
    import numpy as np
    from PIL import Image

    from .looks import grade

    p = need_project(pid)
    if look not in LOOKS:
        raise HTTPException(404)
    src = p.thumbs / f"{Path(cid).name}.jpg"
    if not src.exists():
        raise HTTPException(404)
    out = p.thumbs / f"{Path(cid).name}_look_{look}.jpg"
    if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        img = np.asarray(Image.open(src).convert("RGB"), dtype=np.float32) / 255
        Image.fromarray((grade(img, LOOKS[look]) * 255).astype(np.uint8)).save(out, quality=85)
    return FileResponse(out, headers={"Cache-Control": "max-age=600"})


@app.get("/media/{pid}/renders/{eid}/{name}")
def render_file(pid: str, eid: str, name: str):
    p = need_project(pid)
    f = p.renders / Path(eid).name / Path(name).name
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f)


@app.get("/media/music/{name}")
def music_file(name: str):
    f = MUSIC_DIR / Path(name).name
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f)


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
