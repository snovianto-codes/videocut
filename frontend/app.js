const $app = document.getElementById("app");
const $crumbs = document.getElementById("crumbs");
const $jobsPill = document.getElementById("jobs-pill");
const $toast = document.getElementById("toast");

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtDur = (s) => (s >= 3600 ? `${(s / 3600).toFixed(1)} h` : s >= 60 ? `${Math.round(s / 60)} min` : `${Math.round(s)} s`);
const fmtTime = (iso) => new Date(iso).toLocaleString(undefined, { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });

const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
};

let META = null;
let MUSIC = [];
let PROJECT = null;
const JOBS = new Map();
const DEFAULT_FORM = { duration: 20, platform: "reels", look: "cinematic", music: "", title: "", subtitle: "", slowmo: true, hook: true, letterbox: true, ambient: 0.25, music_section: "auto", ramps: true };
let FORM = { ...DEFAULT_FORM, ...store.get("vc.form", store.get("tc.form", {})) };
if (FORM.platform === "vertical") FORM.platform = "reels";
const platLabel = (k) => { const v = META?.platforms?.[k]; return v ? `${v.name} (${v.aspect})` : k; };

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: opts.body && !(opts.body instanceof FormData) ? { "Content-Type": "application/json" } : {}, ...opts });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch { /* not json */ }
    throw new Error(msg);
  }
  return res.json();
}
const post = (path, body) => api(path, { method: "POST", body: body instanceof FormData ? body : JSON.stringify(body || {}) });

function toast(msg, ms = 3200) {
  $toast.textContent = msg;
  $toast.hidden = false;
  clearTimeout(toast.t);
  toast.t = setTimeout(() => ($toast.hidden = true), ms);
}

// ---------------------------------------------------------------- jobs
function trackJob(job) {
  JOBS.set(job.id, job);
  renderJobsPill();
  pollJobs();
}
const ACTIVE = new Set(["queued", "running", "paused", "cancelling"]);
const isActive = (j) => ACTIVE.has(j.status);
async function pollJobs() {
  if (pollJobs.running) return;
  pollJobs.running = true;
  while ([...JOBS.values()].some(isActive)) {
    await new Promise((r) => setTimeout(r, 900));
    for (const j of [...JOBS.values()]) {
      if (!isActive(j)) continue;
      try {
        const fresh = await api(`/api/jobs/${j.id}`);
        JOBS.set(j.id, fresh);
        if (fresh.status === "done") onJobDone(fresh);
        if (fresh.status === "cancelled") onJobCancelled(fresh);
        if (fresh.status === "error") toast(`Something failed: ${fresh.error.split("\n")[0]}`, 6000);
      } catch { JOBS.delete(j.id); }
    }
    renderJobsPill();
    updateJobCards();
  }
  pollJobs.running = false;
  renderJobsPill();
}
async function onJobDone(job) {
  if (job.kind === "final") toast("Exported! Use “Show in Finder” to grab it.");
  if (job.kind === "edit") toast("Your edit is ready to preview.");
  if (PROJECT && job.project_id === PROJECT.id) {
    await loadProject(PROJECT.id);
    route();
  }
}
async function onJobCancelled(job) {
  toast("Cancelled.");
  JOBS.delete(job.id);
  if (PROJECT && job.project_id === PROJECT.id) {
    await loadProject(PROJECT.id);
    route();
  }
}
function renderJobsPill() {
  const active = [...JOBS.values()].filter(isActive);
  $jobsPill.hidden = !active.length;
  if (active.length) {
    const j = active.find((x) => x.status === "running") || active[0];
    const paused = active.filter((x) => x.status === "paused").length;
    $jobsPill.textContent = `${active.length > 1 ? active.length + " jobs · " : ""}${j.status === "paused" ? "Paused" : j.message} · ${Math.round(j.progress * 100)}%${paused && j.status !== "paused" ? ` · ${paused} paused` : ""}`;
  }
}
const JOB_LABEL = { scan: "Scanning footage", edit: "Making your edit", final: "Exporting full quality", ai: "Claude is watching your clips" };
function jobButtons(j) {
  if (j.status === "cancelling") return `<span class="muted small">Stopping…</span>`;
  if (!isActive(j)) return "";
  return `${j.status === "paused"
    ? `<button class="btn small" data-jobact="resume" data-jid="${j.id}">Resume</button>`
    : `<button class="btn small" data-jobact="pause" data-jid="${j.id}">Pause</button>`}
    <button class="btn small ghost" data-jobact="cancel" data-jid="${j.id}">Cancel</button>`;
}
function jobCard(j) {
  return `<div class="jobcard ${j.status === "paused" ? "paused" : ""}" data-job="${j.id}">
    <div class="row"><b>${JOB_LABEL[j.kind] || j.kind}</b><span class="muted small" data-msg>${j.status === "paused" ? "Paused · " : ""}${esc(j.message)}</span>
      <span class="spacer"></span><span class="row" data-btns>${jobButtons(j)}</span></div>
    <div class="progress"><div data-bar style="width:${Math.round(j.progress * 100)}%"></div></div>
    ${j.error ? `<div class="err">${esc(j.error)}</div>` : ""}
  </div>`;
}
function updateJobCards() {
  document.querySelectorAll("[data-job]").forEach((el) => {
    const j = JOBS.get(el.dataset.job);
    if (!j) return;
    el.querySelector("[data-msg]").textContent = `${j.status === "paused" ? "Paused · " : ""}${j.message}`;
    el.querySelector("[data-bar]").style.width = `${Math.round(j.progress * 100)}%`;
    el.classList.toggle("paused", j.status === "paused");
    if (el.dataset.status !== j.status) {
      el.dataset.status = j.status;
      el.querySelector("[data-btns]").innerHTML = jobButtons(j);
    }
    if (j.status === "error" && !el.querySelector(".err")) el.insertAdjacentHTML("beforeend", `<div class="err">${esc(j.error)}</div>`);
  });
}
document.addEventListener("click", async (ev) => {
  const b = ev.target.closest("[data-jobact]");
  if (!b) return;
  if (b.dataset.jobact === "cancel" && !confirm("Cancel this job? Anything half-made is thrown away.")) return;
  b.disabled = true;
  try {
    const j = await post(`/api/jobs/${b.dataset.jid}/${b.dataset.jobact}`);
    JOBS.set(j.id, j);
    updateJobCards();
    renderJobsPill();
    pollJobs();
  } catch (e) { toast(e.message); b.disabled = false; }
});
const jobsFor = (pred) => [...JOBS.values()].filter((j) => pred(j) && (isActive(j) || j.status === "error"));

// ---------------------------------------------------------------- helpers
function scrub(box) {
  const n = 4;
  const pic = box.querySelector(".pic");
  if (!pic) return;
  box.addEventListener("mousemove", (e) => {
    const r = box.getBoundingClientRect();
    const i = Math.min(n - 1, Math.floor(((e.clientX - r.left) / r.width) * n));
    pic.style.backgroundImage = `url("${pic.dataset.sprite}")`;
    pic.style.backgroundSize = `${n * 100}% 100%`;
    pic.style.backgroundPosition = `${(i * 100) / (n - 1)}% 0`;
  });
  box.addEventListener("mouseleave", () => {
    pic.style.backgroundImage = `url("${pic.dataset.thumb}")`;
    pic.style.backgroundSize = "cover";
    pic.style.backgroundPosition = "center";
  });
}
function picHtml(c) {
  // the clip at its real shape inside a 16:9 box: portrait and 4:3 clips aren't stretched
  const fit = c.width / c.height <= 16 / 9 ? "fit-h" : "fit-w";
  return `<div class="pic ${fit}" style="aspect-ratio:${c.width}/${c.height};background-image:url('${thumbUrl(c.id)}')"
    data-thumb="${thumbUrl(c.id)}" data-sprite="${spriteUrl(c.id)}"></div>`;
}
const thumbVer = (cid) => { const c = PROJECT.clips[cid]; return c ? `?v=${c.rotate_fix || 0}-${c.profile}` : ""; };
const thumbUrl = (cid) => `/media/${PROJECT.id}/thumbs/${cid}.jpg${thumbVer(cid)}`;
const spriteUrl = (cid) => `/media/${PROJECT.id}/thumbs/${cid}_sprite.jpg${thumbVer(cid)}`;
const isLog = (p) => p && p !== "rec709";
const profileName = (p) => (META?.profiles?.[p] || p);
function aspectLabel(w, h) {
  const r = w / h;
  const known = [[16 / 9, "16:9"], [4 / 3, "4:3"], [1, "1:1"], [3 / 4, "3:4"], [9 / 16, "9:16"], [2.39, "2.39:1"], [21 / 9, "21:9"]];
  const hit = known.find(([k]) => Math.abs(r - k) / k < 0.03);
  return hit ? hit[1] : `${w}×${h}`;
}
const ORIENT_ICON = { portrait: "▯", landscape: "▭", square: "◻" };
const orientOf = (c) => (c.height > c.width ? "portrait" : c.width > c.height ? "landscape" : "square");
function orientBadge(c) {
  const o = orientOf(c);
  return `<span class="orient ${o}" title="${o === "portrait" ? "Portrait (vertical)" : o === "square" ? "Square" : "Landscape (horizontal)"} · ${c.width}×${c.height}">${ORIENT_ICON[o] || "◻"} ${o === "portrait" ? "Portrait" : o === "square" ? "Square" : "Landscape"} ${aspectLabel(c.width, c.height)}</span>`;
}
function orientCounts(clips) {
  const n = { landscape: 0, portrait: 0, square: 0 };
  clips.forEach((c) => n[orientOf(c)]++);
  return n;
}
function orientSummary(clips, asBadges) {
  const n = orientCounts(clips);
  return ["landscape", "portrait", "square"].filter((k) => n[k]).map((k) => asBadges
    ? `<span class="badge ${k === "portrait" ? "portrait" : ""}" title="${k[0].toUpperCase() + k.slice(1)} clips">${ORIENT_ICON[k]} ${n[k]} ${k}</span>`
    : `${ORIENT_ICON[k]} ${n[k]} ${k}`).join(asBadges ? "" : " · ");
}

async function loadProject(pid) {
  PROJECT = await api(`/api/projects/${pid}`);
  for (const j of PROJECT.jobs || []) if (!JOBS.has(j.id)) trackJob(j);
  return PROJECT;
}

// ---------------------------------------------------------------- views
async function viewHome() {
  $crumbs.innerHTML = "";
  const recent = await api("/api/projects");
  $app.innerHTML = `
    <section class="hero">
      <h1>Dump your trip footage.<br>Get reels back.</h1>
      <p>Point VideoCut at the folder from your Sony, Fujifilm or DJI. It groups clips by day, place or theme, finds the best moments, cuts them to the beat, grades them, and reframes for Reels, TikTok, Shorts or YouTube.</p>
      <div class="open-box">
        <button class="btn primary" id="pick">Choose folder…</button>
        <input type="text" id="path" placeholder="…or paste a folder path, e.g. ~/Movies/Bali 2026">
        <button class="btn" id="open">Open</button>
      </div>
    </section>
    ${recent.length ? `<h3>Recent trips</h3><div class="recent" style="margin:10px 0 30px">${recent.map((r) => `
      <a href="#/p/${r.id}"><span><b>${esc(r.name)}</b> <span class="muted small">${esc(r.folder)}</span></span><span class="muted">${r.clips} clips · ${r.groups} groups</span></a>`).join("")}</div>` : ""}
    <div class="features">
      ${[
        ["Auto grouping", "Splits the trip by day, time gaps and GPS, or by theme (beach, food, temples) with Claude."],
        ["Log footage handled", "Detects S-Log3, F-Log, D-Log and HLG and converts them using the camera’s own curves."],
        ["Best-moment finder", "Scores every second for sharpness, colour, motion, shake, exposure, faces and sound."],
        ["Beat-synced cuts", "Drop music in and cuts land on the beat, with a hook shot first and a calm ending."],
        ["Smart reframing", "Follows the subject for 9:16, slow-pans across landscapes, blur-fills portrait clips."],
        ["Cinematic looks", "Teal & orange, film, golden hour, moody, B&W, with grain, glow, bars and 24 fps."],
      ].map(([b, s]) => `<div class="feature"><b>${b}</b><span>${s}</span></div>`).join("")}
    </div>`;
  const openPath = async (folder) => {
    if (!folder) return;
    try {
      const r = await post("/api/projects/open", { folder });
      if (r.job) trackJob(r.job);
      location.hash = `#/p/${r.project_id}`;
    } catch (e) { toast(e.message); }
  };
  document.getElementById("pick").onclick = async (e) => {
    const btn = e.currentTarget;
    btn.disabled = true;
    btn.textContent = "Choose in the dialog…";
    try {
      const r = await post("/api/pick-folder");
      if (r.folder) return openPath(r.folder);
      if (r.error) {
        toast(r.error, 8000);
        document.getElementById("path").focus();
      }
    } catch (err) {
      toast(err.message, 6000);
    } finally {
      btn.disabled = false;
      btn.textContent = "Choose folder…";
    }
  };
  document.getElementById("open").onclick = () => openPath(document.getElementById("path").value.trim());
  document.getElementById("path").onkeydown = (e) => e.key === "Enter" && openPath(e.target.value.trim());
}

function viewProject() {
  const p = PROJECT;
  $crumbs.innerHTML = `<span>/</span><a href="#/p/${p.id}">${esc(p.name)}</a>`;
  const clips = Object.values(p.clips);
  const footage = clips.reduce((a, c) => a + c.duration, 0);
  const cams = [...new Set(clips.map((c) => c.camera))];
  const days = new Set(clips.map((c) => c.captured.slice(0, 10))).size;
  const scanJobs = jobsFor((j) => j.project_id === p.id && (j.kind === "scan" || j.kind === "ai"));
  const editJobs = jobsFor((j) => j.project_id === p.id && j.kind === "edit");
  const selected = new Set();
  $app.innerHTML = `
    <div class="proj-head">
      <div><h1>${esc(p.name)}</h1><div class="muted small">${esc(p.folder)}</div></div>
      <span class="spacer"></span>
      <div class="stats">
        <div><b>${clips.length}</b>clips</div><div><b>${fmtDur(footage)}</b>footage</div>
        <div><b>${days}</b>days</div><div><b>${p.groups.length}</b>groups</div><div><b>${cams.length}</b>cameras</div>
      </div>
    </div>
    ${scanJobs.map(jobCard).join("")}
    ${clips.length ? `
    <div class="toolbar">
      <span class="muted">Group by</span>
      <div class="seg">
        <button data-mode="moment" class="${p.group_mode !== "ai" ? "on" : ""}">Day, time & place</button>
        <button data-mode="ai" class="${p.group_mode === "ai" ? "on" : ""}" ${META.ai ? "" : "disabled title='Set ANTHROPIC_API_KEY to enable'"}>Theme (Claude)</button>
      </div>
      ${p.group_mode !== "ai" ? `<label class="slider">New group after <input type="range" id="gap" min="0.5" max="12" step="0.5" value="${p.settings.gap_hours ?? 3}"> <span id="gapv">${p.settings.gap_hours ?? 3} h</span> gap</label>` : ""}
      ${META.ai ? `<button class="btn small" id="ai">${p.has_ai_groups ? "Re-run" : "Find"} themes with Claude</button>` : ""}
      <span class="spacer"></span>
      <button class="btn small" id="merge" disabled>Merge selected</button>
      <button class="btn small ghost" id="rescan">Rescan folder</button>
      <button class="btn small primary" id="all">Make reels for every group</button>
    </div>
    ${editJobs.length ? `<div style="display:grid;gap:8px;margin-bottom:16px">${editJobs.map(jobCard).join("")}</div>` : ""}
    <div class="groups">${p.groups.map((g) => groupCard(g)).join("")}</div>` : (scanJobs.length ? "" : `<div class="empty">No videos found yet. <button class="btn small" id="rescan">Scan folder</button></div>`)}`;

  $app.querySelectorAll(".frame, .cover").forEach(scrub);
  $app.querySelectorAll(".gcard .cover").forEach((el) => (el.onclick = () => (location.hash = `#/p/${p.id}/g/${el.dataset.gid}`)));
  $app.querySelectorAll(".gcard .pick").forEach((cb) => (cb.onchange = () => {
    cb.checked ? selected.add(cb.value) : selected.delete(cb.value);
    document.getElementById("merge").disabled = selected.size < 2;
  }));
  $app.querySelectorAll("[data-mode]").forEach((b) => (b.onclick = async () => {
    if (b.dataset.mode === "ai" && !p.has_ai_groups) return startAI();
    await post(`/api/projects/${p.id}/group-mode/${b.dataset.mode}`);
    await loadProject(p.id); viewProject();
  }));
  const gap = document.getElementById("gap");
  if (gap) {
    gap.oninput = () => (document.getElementById("gapv").textContent = `${gap.value} h`);
    gap.onchange = async () => { await post(`/api/projects/${p.id}/regroup`, { gap_hours: +gap.value, km: p.settings.km ?? 10 }); await loadProject(p.id); viewProject(); };
  }
  const startAI = async () => {
    try { trackJob(await post(`/api/projects/${p.id}/ai-themes`)); viewProject(); } catch (e) { toast(e.message); }
  };
  document.getElementById("ai")?.addEventListener("click", startAI);
  document.getElementById("rescan")?.addEventListener("click", async () => { trackJob(await post(`/api/projects/${p.id}/rescan`)); viewProject(); });
  document.getElementById("merge")?.addEventListener("click", async () => {
    await post(`/api/projects/${p.id}/groups/merge`, { group_ids: [...selected] });
    await loadProject(p.id); viewProject();
  });
  document.getElementById("all")?.addEventListener("click", async () => {
    const look = p.group_mode === "ai" ? "auto" : FORM.look;
    if (!confirm(`Make a ${FORM.duration}s ${platLabel(FORM.platform)} video for all ${p.groups.length} groups${look === "auto" ? " (each with Claude’s suggested look)" : ` in the ${META.looks.find((l) => l.key === FORM.look).name} look`}?`)) return;
    const jobs = await post(`/api/projects/${p.id}/edit-all`, { ...formBody(), look });
    jobs.forEach(trackJob);
    viewProject();
  });
}

function groupCard(g) {
  const p = PROJECT;
  const cover = g.cover || g.clip_ids[0];
  const clips = g.clip_ids.map((i) => p.clips[i]).filter(Boolean);
  const logs = [...new Set(clips.filter((c) => isLog(c.profile)).map((c) => profileName(c.profile)))];
  const made = (p.edits[g.id] || []).length;
  const days = g.days.length > 1 ? `Days ${g.days[0]}–${g.days.at(-1)}` : `Day ${g.days[0]}`;
  return `<div class="gcard">
    <div class="cover" data-gid="${g.id}">${p.clips[cover] ? picHtml(p.clips[cover]) : ""}
      <span class="count">${g.clip_ids.length} clips · ${fmtDur(g.footage_seconds)}</span>
    </div>
    <input type="checkbox" class="pick" value="${g.id}" title="Select to merge">
    <div class="body">
      <a class="title" href="#/p/${p.id}/g/${g.id}">${esc(g.title)}</a>
      ${g.description ? `<div class="desc">${esc(g.description)}</div>` : `<div class="desc">${days} · ${fmtTime(g.start)}</div>`}
      <div class="badges">
        ${g.cameras.map((c) => `<span class="badge">${esc(c)}</span>`).join("")}
        ${orientSummary(clips, true)}
        ${g.has_drone ? `<span class="badge drone">aerial</span>` : ""}
        ${logs.map((l) => `<span class="badge log">${esc(l)}</span>`).join("")}
        ${g.suggested_look ? `<span class="badge accent">${esc(META.looks.find((l) => l.key === g.suggested_look)?.name || g.suggested_look)}</span>` : ""}
        ${g.music_mood ? `<span class="badge">♪ ${esc(g.music_mood)}</span>` : ""}
        ${made ? `<span class="badge made">${made} edit${made > 1 ? "s" : ""}</span>` : ""}
      </div>
    </div>
  </div>`;
}

function formBody() {
  return { ...FORM, music: FORM.music || null, duration: +FORM.duration, ambient: FORM.music ? +FORM.ambient : null };
}

function viewGroup(gid) {
  const p = PROJECT;
  const g = p.groups.find((x) => x.id === gid) || null;
  if (!g) { location.hash = `#/p/${p.id}`; return; }
  $crumbs.innerHTML = `<span>/</span><a href="#/p/${p.id}">${esc(p.name)}</a><span>/</span><span>${esc(g.title)}</span>`;
  const clips = g.clip_ids.map((i) => p.clips[i]).filter(Boolean);
  const edits = p.edits[g.id] || [];
  const jobs = jobsFor((j) => j.project_id === p.id && (j.group_id === g.id || edits.some((e) => e.id === j.edit_id)));
  if (g.suggested_look && !store.get(`vc.look.${g.id}`, null)) FORM.look = g.suggested_look;
  const cover = g.cover || g.clip_ids[0];
  const usable = clips.filter((c) => !c.excluded);

  $app.innerHTML = `
    <div style="margin-bottom:18px">
      <input class="title-edit" id="gtitle" value="${esc(g.title)}" aria-label="Group title">
      <div class="muted">${fmtTime(g.start)} → ${fmtTime(g.end)} · ${usable.length} clips (${orientSummary(usable, false)}) · ${fmtDur(g.footage_seconds)} of footage
        ${g.description ? ` · ${esc(g.description)}` : ""}${g.music_mood ? ` · music idea: ${esc(g.music_mood)}` : ""}</div>
    </div>
    <div class="layout">
      <div>
        <div class="panel">
          <div class="row" style="margin-bottom:12px"><h2>Edits</h2><span class="spacer"></span><span class="muted small">Previews are half resolution. Export renders full quality.</span></div>
          <div class="edits">
            ${jobs.map(jobCard).join("")}
            ${edits.length ? edits.map(editCard).join("") : jobs.length ? "" : `<div class="empty">Pick a length, platform and look on the right, then press <b>Make my edit</b>.</div>`}
          </div>
        </div>
        <div class="panel">
          <div class="row" style="margin-bottom:12px"><h2>Clips</h2><span class="spacer"></span><span class="muted small">Hover to scrub. Hide clips you never want used. Fix the colour profile if it was misdetected.</span></div>
          <div class="clips">${clips.map(clipCard).join("")}</div>
        </div>
      </div>
      <aside class="panel sticky" id="form">${formHtml(cover)}</aside>
    </div>`;

  $app.querySelectorAll(".frame, .cover").forEach(scrub);
  document.getElementById("gtitle").onchange = async (e) => {
    await api(`/api/projects/${p.id}/groups/${g.id}`, { method: "PATCH", body: JSON.stringify({ title: e.target.value }) });
    await loadProject(p.id);
    $crumbs.lastElementChild.textContent = e.target.value;
  };
  bindForm(g, cover);
  bindClips();
  bindEdits(g);
}

function formHtml(cover) {
  const lens = [10, 15, 20, 30, 45, 60];
  const shape = (w, h) => { const s = 26 / Math.max(w, h); return `<span class="shape" style="width:${w * s}px;height:${h * s}px"></span>`; };
  return `
    <h2 style="margin-bottom:14px">Make a reel</h2>
    <div class="field"><label>Length</label>
      <div class="chips">${lens.map((l) => `<button class="chip ${+FORM.duration === l ? "on" : ""}" data-len="${l}">${l}s</button>`).join("")}
        <input type="number" id="len" min="5" max="180" value="${FORM.duration}" style="width:70px" aria-label="Custom length in seconds"></div>
    </div>
    <div class="field"><label>Where it’s going</label>
      <div class="plat">${Object.entries(META.platforms).filter(([, v]) => !v.hidden).map(([k, v]) => `<button data-plat="${k}" class="${FORM.platform === k ? "on" : ""}">${shape(v.w, v.h)}<span>${esc(v.name)}</span><span class="aspect">${esc(v.aspect)}</span></button>`).join("")}</div>
    </div>
    <div class="field"><label>Look</label>
      <div class="looks">${META.looks.map((l) => `<button class="look ${FORM.look === l.key ? "on" : ""}" data-look="${l.key}" title="${esc(l.blurb)}">
        <div class="sw" style="background-image:url('/media/${PROJECT.id}/look/${cover}/${l.key}.jpg${thumbVer(cover)}')"></div><div class="lbl">${esc(l.name)}</div></button>`).join("")}</div>
      <div class="hint" id="lookblurb">${esc(META.looks.find((l) => l.key === FORM.look)?.blurb || "")}</div>
    </div>
    <div class="field"><label>Music</label>
      <div class="row"><select id="music" style="flex:1"><option value="">No music (natural sound)</option>
        ${MUSIC.map((m) => `<option ${FORM.music === m.name ? "selected" : ""}>${esc(m.name)}</option>`).join("")}</select>
        <label class="btn small">Add…<input type="file" id="upload" accept="audio/*" hidden></label></div>
      ${FORM.music ? `
      <div class="seg" role="group" aria-label="Part of the song">
        <button data-section="auto" class="${FORM.music_section !== "start" ? "on" : ""}">Best part (build-up → drop)</button>
        <button data-section="start" class="${FORM.music_section === "start" ? "on" : ""}">From the start</button>
      </div>
      <label class="check"><input type="checkbox" id="ramps" ${FORM.ramps !== false ? "checked" : ""}> Speed ramp, zoom punch & flash on the drop</label>
      <label class="slider">Camera sound <input type="range" id="ambient" min="0" max="1" step="0.05" value="${FORM.ambient}"></label>` : ""}
      <div class="hint">${FORM.music ? "Cuts follow the bars and energy of the song and land exactly on the drop." : "Pick a track and the edit is cut to its beats, sections and drop."} Use music you have the rights to post.</div>
    </div>
    <div class="field"><label>Title card</label>
      <input type="text" id="title" placeholder="e.g. BALI (leave empty for none)" value="${esc(FORM.title)}">
      <input type="text" id="subtitle" placeholder="Subtitle, e.g. September 2026" value="${esc(FORM.subtitle)}">
    </div>
    <div class="field"><label>Extras</label>
      <label class="check"><input type="checkbox" id="hook" ${FORM.hook ? "checked" : ""}> Open with the most eye-catching shot</label>
      <label class="check"><input type="checkbox" id="slowmo" ${FORM.slowmo ? "checked" : ""}> Slow-motion for 50/60/120 fps clips</label>
      <label class="check"><input type="checkbox" id="letterbox" ${FORM.letterbox ? "checked" : ""}> Widescreen bars (16:9 cinematic)</label>
    </div>
    <button class="btn primary" id="go" style="width:100%;justify-content:center;padding:12px">Make my edit</button>`;
}

function bindForm(g, cover) {
  const save = () => store.set("vc.form", FORM);
  const rerender = () => { document.getElementById("form").innerHTML = formHtml(cover); bindForm(g, cover); };
  document.querySelectorAll("[data-len]").forEach((b) => (b.onclick = () => { FORM.duration = +b.dataset.len; save(); rerender(); }));
  document.getElementById("len").onchange = (e) => { FORM.duration = Math.max(5, Math.min(180, +e.target.value || 20)); save(); rerender(); };
  document.querySelectorAll("[data-plat]").forEach((b) => (b.onclick = () => { FORM.platform = b.dataset.plat; save(); rerender(); }));
  document.querySelectorAll("[data-look]").forEach((b) => (b.onclick = () => { FORM.look = b.dataset.look; store.set(`vc.look.${g.id}`, FORM.look); save(); rerender(); }));
  document.getElementById("music").onchange = (e) => { FORM.music = e.target.value; save(); rerender(); };
  document.getElementById("ambient")?.addEventListener("input", (e) => { FORM.ambient = +e.target.value; save(); });
  document.getElementById("ramps")?.addEventListener("change", (e) => { FORM.ramps = e.target.checked; save(); });
  document.querySelectorAll("[data-section]").forEach((b) => (b.onclick = () => { FORM.music_section = b.dataset.section; save(); rerender(); }));
  ["title", "subtitle"].forEach((k) => (document.getElementById(k).oninput = (e) => { FORM[k] = e.target.value; save(); }));
  ["hook", "slowmo", "letterbox"].forEach((k) => (document.getElementById(k).onchange = (e) => { FORM[k] = e.target.checked; save(); }));
  document.getElementById("upload").onchange = async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const fd = new FormData();
    fd.append("file", f);
    try {
      const r = await post("/api/music", fd);
      MUSIC = await api("/api/music");
      FORM.music = r.name; save(); rerender();
      toast(`Added ${r.name}`);
    } catch (err) { toast(err.message); }
  };
  document.getElementById("go").onclick = () => startEdit(g.id, formBody());
}

async function startEdit(gid, body) {
  try {
    trackJob(await post(`/api/projects/${PROJECT.id}/groups/${gid}/edit`, body));
    viewGroup(gid);
    window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (e) { toast(e.message); }
}

function clipCard(c) {
  const opts = Object.entries(META.profiles).map(([k, v]) => `<option value="${k}" ${c.profile === k ? "selected" : ""}>${esc(v)}</option>`).join("");
  return `<div class="clip ${c.excluded ? "off" : ""}" data-cid="${c.id}">
    <div class="frame">${picHtml(c)}${orientBadge(c)}</div>
    ${c.tags?.scene ? `<span class="tag">${esc(c.tags.scene)}</span>` : ""}
    <div class="tools">
      <button data-rotate="-90" title="Rotate 90° left (fix a sideways clip)" aria-label="Rotate left">↺</button>
      <button data-rotate="90" title="Rotate 90° right (fix a sideways clip)" aria-label="Rotate right">↻</button>
      <button data-exclude="${c.excluded ? "0" : "1"}">${c.excluded ? "Use" : "Hide"}</button>
    </div>
    <div class="meta">
      <span class="name" title="${esc(c.path)}">${esc(c.name)}</span>
      <span class="muted">${esc(c.camera)} · ${Math.round(c.duration)}s · ${Math.round(c.fps)}p${c.rotate_fix ? ` · rotated ${c.rotate_fix}°` : ""}</span>
      <select data-profile aria-label="Colour profile">${opts}</select>
    </div>
  </div>`;
}

function bindClips() {
  document.querySelectorAll(".clip").forEach((el) => {
    const cid = el.dataset.cid;
    el.querySelectorAll("[data-rotate]").forEach((b) => (b.onclick = async () => {
      el.querySelectorAll(".tools button").forEach((x) => (x.disabled = true));
      el.classList.add("busy");
      try {
        await api(`/api/projects/${PROJECT.id}/clips/${cid}`, { method: "PATCH", body: JSON.stringify({ rotate: +b.dataset.rotate }) });
        await loadProject(PROJECT.id); route();
      } catch (err) { toast(err.message); el.classList.remove("busy"); }
    }));
    el.querySelector("[data-exclude]").onclick = async (e) => {
      await api(`/api/projects/${PROJECT.id}/clips/${cid}`, { method: "PATCH", body: JSON.stringify({ excluded: e.target.dataset.exclude === "1" }) });
      await loadProject(PROJECT.id); route();
    };
    el.querySelector("[data-profile]").onchange = async (e) => {
      toast("Re-grading thumbnails…");
      await api(`/api/projects/${PROJECT.id}/clips/${cid}`, { method: "PATCH", body: JSON.stringify({ profile: e.target.value }) });
      await loadProject(PROJECT.id); route();
    };
  });
}

function musicMap(e) {
  const m = e.music?.map;
  if (!m || !m.beats?.length) return "";
  const D = e.plan.duration;
  const X = (t) => ((t / D) * 1000).toFixed(1);
  const Y = (v) => (52 - v * 42).toFixed(1);
  const pts = m.beats.map((t, i) => `${X(t)},${Y(m.energy[i] ?? 0)}`);
  const area = `M0,52 L${pts.join(" L")} L${X(m.beats.at(-1))},52 Z`;
  const cuts = e.plan.shots.slice(1).map((s) => `<line x1="${X(s.timeline[0])}" x2="${X(s.timeline[0])}" y1="0" y2="8" class="mm-cut"/>`).join("");
  const bars = m.bars.map((t) => `<line x1="${X(t)}" x2="${X(t)}" y1="48" y2="56" class="mm-bar"/>`).join("");
  const secs = m.sections.map((t) => `<line x1="${X(t)}" x2="${X(t)}" y1="8" y2="56" class="mm-sec"/>`).join("");
  const drops = m.drops.map((t) => `<line x1="${X(t)}" x2="${X(t)}" y1="4" y2="56" class="mm-drop"/><text x="${+X(t) + 6}" y="20" class="mm-label">DROP</text>`).join("");
  const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  return `<div class="mm" title="Song energy with cuts (top ticks), bars (bottom ticks), section changes and the drop. Click to jump.">
    <svg viewBox="0 0 1000 60" preserveAspectRatio="none" data-mm data-dur="${D}">
      <path d="${area}" class="mm-area"/><polyline points="${pts.join(" ")}" class="mm-line"/>
      ${bars}${secs}${cuts}${drops}<line x1="0" x2="0" y1="0" y2="60" class="mm-head" data-head/>
    </svg>
    <div class="muted small">Using ${fmt(m.start)}–${fmt(m.end)} of the track${m.drops.length ? " · cuts build up into the drop" : ""}</div>
  </div>`;
}

function safeZones(platform) {
  const zones = META.platforms[platform]?.safe;
  if (!zones) return "";
  return `<div class="safe">${zones.map(([l, t, w, h]) => `<div style="left:${l}%;top:${t}%;width:${w}%;height:${h}%"></div>`).join("")}</div>`;
}

function editCard(e) {
  const plan = e.plan;
  const look = META.looks.find((l) => l.key === plan.look)?.name || plan.look;
  const vertical = plan.height > plan.width;
  const w = vertical ? 300 : 560;
  const src = `/media/${PROJECT.id}/renders/${e.id}/${e.preview}`;
  return `<div class="edit" data-eid="${e.id}">
    <div class="player" style="width:${w}px;max-width:100%">
      <video src="${src}" poster="/media/${PROJECT.id}/renders/${e.id}/${e.thumbs[0] || ""}" controls playsinline preload="metadata" style="width:100%"></video>
      ${safeZones(plan.platform)}
    </div>
    <div class="edit-info">
      <div><b>${esc(look)}</b> · ${esc(platLabel(plan.platform))} · ${plan.duration.toFixed(1)}s · ${plan.shots.length} shots · ${plan.fps} fps</div>
      <div class="muted small">${e.music ? `♪ ${esc(e.music.path.split("/").pop())} · ${e.music.tempo} BPM` : "Natural sound"} · made ${fmtTime(e.created)}</div>
      ${musicMap(e)}
      <div class="storyboard">${plan.shots.map((s, i) => `
        <div class="shot" data-seek="${s.timeline[0]}" title="${esc(s.name)} · ${esc(s.camera)}${s.speed < 1 ? ` · ${Math.round(1 / s.speed * 10) / 10}x slow-mo` : ""} · ${s.crop.why ? "reframe: " + s.crop.why : s.crop.mode}">
          ${s.transition_in.type !== "cut" ? `<span class="tr">${esc(s.transition_in.type)}</span>` : ""}
          <img src="/media/${PROJECT.id}/renders/${e.id}/${e.thumbs[i] || ""}" alt="">
          <button class="x" data-drop="${s.clip_id}" title="Never use this clip here">✕</button>
          ${s.music?.drop ? `<span class="tr drop">drop</span>` : ""}
          <div class="t"><span>${s.timeline[0].toFixed(1)}s</span><span>${s.ramp ? "ramp" : s.speed < 1 ? "slo-mo" : ""}</span></div>
        </div>`).join("")}</div>
      <div class="row">
        ${e.final ? `<button class="btn primary small" data-reveal="${esc(e.final)}">Show in Finder</button>` : `<button class="btn primary small" data-final>Export full quality</button>`}
        <button class="btn small" data-again>Try another cut</button>
        ${META.platforms[plan.platform]?.safe ? `<label class="check small"><input type="checkbox" data-safe> Show where ${esc(META.platforms[plan.platform].hidden ? "the app" : META.platforms[plan.platform].name)} covers the video</label>` : ""}
        <span class="spacer"></span>
        <button class="btn small ghost" data-del>Delete</button>
      </div>
      ${e.final ? `<div class="muted small">Saved to ${esc(e.final)}</div>` : ""}
    </div>
  </div>`;
}

function bindEdits(g) {
  document.querySelectorAll(".edit").forEach((el) => {
    const e = (PROJECT.edits[g.id] || []).find((x) => x.id === el.dataset.eid);
    const video = el.querySelector("video");
    const mm = el.querySelector("[data-mm]");
    if (mm) {
      const head = mm.querySelector("[data-head]");
      const D = +mm.dataset.dur;
      video.addEventListener("timeupdate", () => {
        const x = Math.min(1000, (video.currentTime / D) * 1000);
        head.setAttribute("x1", x); head.setAttribute("x2", x);
      });
      mm.addEventListener("click", (ev) => {
        const r = mm.getBoundingClientRect();
        video.currentTime = ((ev.clientX - r.left) / r.width) * D;
        video.play();
      });
    }
    el.querySelectorAll("[data-seek]").forEach((s) => (s.onclick = (ev) => {
      if (ev.target.dataset.drop) return;
      video.currentTime = +s.dataset.seek + 0.05; video.play();
    }));
    el.querySelectorAll("[data-drop]").forEach((b) => (b.onclick = () => {
      const s = e.settings;
      startEdit(g.id, { ...s, exclude_clips: [...(s.exclude_clips || []), b.dataset.drop] });
    }));
    el.querySelector("[data-again]").onclick = () => startEdit(g.id, { ...e.settings, seed: (e.settings.seed || 0) + 1 + Math.floor(Math.random() * 1000) });
    el.querySelector("[data-final]")?.addEventListener("click", async () => {
      trackJob(await post(`/api/projects/${PROJECT.id}/edits/${e.id}/final`));
      viewGroup(g.id);
    });
    el.querySelector("[data-reveal]")?.addEventListener("click", (ev) => post("/api/reveal", { path: ev.target.dataset.reveal }).catch((err) => toast(err.message)));
    el.querySelector("[data-safe]")?.addEventListener("change", (ev) => el.querySelector(".safe").classList.toggle("on", ev.target.checked));
    el.querySelector("[data-del]").onclick = async () => {
      if (!confirm("Delete this edit preview?")) return;
      await api(`/api/projects/${PROJECT.id}/edits/${e.id}`, { method: "DELETE" });
      await loadProject(PROJECT.id); route();
    };
  });
}

// ---------------------------------------------------------------- router
async function route() {
  const parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  try {
    if (!META) [META, MUSIC] = await Promise.all([api("/api/meta"), api("/api/music")]);
    if (parts[0] === "p" && parts[1]) {
      if (!PROJECT || PROJECT.id !== parts[1]) await loadProject(parts[1]);
      if (parts[2] === "g" && parts[3]) viewGroup(parts[3]);
      else viewProject();
    } else {
      PROJECT = null;
      await viewHome();
    }
  } catch (e) {
    $app.innerHTML = `<div class="empty">${esc(e.message)}<br><br><a class="btn" href="#/">Back home</a></div>`;
  }
}
window.addEventListener("hashchange", route);
route();

const UI_VERSION = new URL(import.meta.url).searchParams.get("v") || "dev";
document.getElementById("ver").textContent = `v${UI_VERSION}`;
document.getElementById("update").onclick = () => location.reload();
async function checkVersion() {
  try {
    const { ui } = await api("/api/version");
    const stale = ui !== UI_VERSION;
    document.getElementById("update").hidden = !stale;
    document.getElementById("ver").classList.toggle("stale", stale);
    document.getElementById("ver").title = stale ? `This window has v${UI_VERSION}; the server has v${ui}` : "You have the latest version";
  } catch { /* server restarting */ }
}
checkVersion();
setInterval(checkVersion, 30000);
window.addEventListener("focus", checkVersion);
