"""Projects (one per footage folder) persisted as JSON, plus a tiny background job runner."""
from __future__ import annotations

import hashlib
import json
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import control
from .config import DATA_DIR
from .control import Cancelled, JobControl

ACTIVE = ("queued", "running", "paused", "cancelling")
PROJECTS_DIR = DATA_DIR / "projects"
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)


class Project:
    def __init__(self, pid: str):
        self.id = pid
        self.dir = PROJECTS_DIR / pid
        self.thumbs = self.dir / "thumbs"
        self.renders = self.dir / "renders"
        for d in (self.dir, self.thumbs, self.renders):
            d.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "project.json"
        self.lock = threading.RLock()
        self.data = json.loads(self.file.read_text()) if self.file.exists() else {}

    @staticmethod
    def id_for(folder: Path) -> str:
        return hashlib.sha1(str(folder.resolve()).encode()).hexdigest()[:10]

    def save(self):
        with self.lock:
            tmp = self.file.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=1))
            tmp.replace(self.file)

    def update(self, fn: Callable[[dict], None]):
        with self.lock:
            fn(self.data)
            self.save()

    @property
    def clips(self) -> dict[str, dict]:
        return self.data.get("clips", {})

    def groups(self) -> list[dict]:
        mode = self.data.get("group_mode", "moment")
        key = "ai_groups" if mode == "ai" and self.data.get("ai_groups") else "groups"
        return self.data.get(key, [])

    def group(self, gid: str) -> dict | None:
        for g in self.data.get("groups", []) + self.data.get("ai_groups", []):
            if g["id"] == gid:
                return g
        return None


_projects: dict[str, Project] = {}
_plock = threading.Lock()


def get_project(pid: str) -> Project | None:
    with _plock:
        if pid not in _projects:
            if not (PROJECTS_DIR / pid / "project.json").exists():
                return None
            _projects[pid] = Project(pid)
        return _projects[pid]


def open_project(folder: Path) -> Project:
    pid = Project.id_for(folder)
    with _plock:
        if pid not in _projects:
            _projects[pid] = Project(pid)
        p = _projects[pid]
    if not p.data:
        p.update(lambda d: d.update({"id": pid, "folder": str(folder.resolve()), "name": folder.name,
                                     "created": datetime.now().isoformat(timespec="seconds"),
                                     "clips": {}, "groups": [], "edits": {}, "settings": {"gap_hours": 3, "km": 10}}))
    return p


def list_projects() -> list[dict]:
    out = []
    for f in PROJECTS_DIR.glob("*/project.json"):
        try:
            d = json.loads(f.read_text())
            out.append({"id": d["id"], "name": d["name"], "folder": d["folder"], "clips": len(d.get("clips", {})),
                        "groups": len(d.get("groups", [])), "updated": f.stat().st_mtime})
        except (json.JSONDecodeError, KeyError):
            continue
    return sorted(out, key=lambda x: -x["updated"])


class Jobs:
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.jobs: dict[str, dict] = {}
        self.controls: dict[str, JobControl] = {}
        self.lock = threading.Lock()

    def start(self, kind: str, fn: Callable[[Callable[[float, str], None]], object], **meta) -> dict:
        jid = uuid.uuid4().hex[:10]
        job = {"id": jid, "kind": kind, "status": "queued", "progress": 0.0, "message": "Queued", "result": None,
               "error": None, "created": datetime.now().isoformat(timespec="seconds"), **meta}
        ctl = JobControl()
        with self.lock:
            self.jobs[jid] = job
            self.controls[jid] = ctl

        def progress(p: float, msg: str):
            control.checkpoint()
            job.update(progress=round(min(1.0, max(0.0, p)), 3), message=msg)

        def run():
            token = control.CURRENT.set(ctl)
            try:
                ctl.checkpoint()
                job.update(status="running", started=True)
                job["result"] = fn(progress)
                job.update(status="done", progress=1.0)
            except Cancelled:
                job.update(status="cancelled", message="Cancelled")
            except Exception as e:  # surfaced to the UI
                if ctl.cancelled.is_set():
                    job.update(status="cancelled", message="Cancelled")
                else:
                    traceback.print_exc()
                    job.update(status="error", error=str(e)[-1500:], message="Failed")
            finally:
                control.CURRENT.reset(token)

        self.pool.submit(run)
        return job

    def get(self, jid: str) -> dict | None:
        return self.jobs.get(jid)

    def active(self) -> list[dict]:
        return [j for j in self.jobs.values() if j["status"] in ACTIVE]

    def pause(self, jid: str) -> dict:
        job, ctl = self.jobs[jid], self.controls[jid]
        if job["status"] in ("queued", "running"):
            ctl.pause()
            job["status"] = "paused"
        return job

    def resume(self, jid: str) -> dict:
        job, ctl = self.jobs[jid], self.controls[jid]
        if job["status"] == "paused":
            ctl.resume()
            job["status"] = "running" if job.get("started") else "queued"
        return job

    def cancel(self, jid: str) -> dict:
        job, ctl = self.jobs[jid], self.controls[jid]
        if job["status"] in ACTIVE:
            ctl.cancel()
            job.update(status="cancelling", message="Cancelling…")
        return job


jobs = Jobs()
