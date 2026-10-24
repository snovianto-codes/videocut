"""Render an edit plan with ffmpeg: graded/reframed shot segments, then one assembly pass."""
from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import control
from .config import FFMPEG, FONT_FALLBACK, FONT_TITLE
from .looks import LOOKS, lut_path
from .media import rotate_filter


def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def _esc(path: Path | str) -> str:
    return str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def run(cmd: list[str]):
    r = control.run(cmd, text=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n" + " ".join(cmd[:6]) + " ...\n" + r.stderr[-2500:])


def color_in(clip_space: str, rng: str) -> str:
    m = "bt2020" if clip_space.startswith("bt2020") else "bt709"
    return f"in_color_matrix={m}:in_range={'pc' if rng == 'pc' else 'tv'}"


def shot_video_filter(shot: dict, clip: dict, W: int, H: int, fps: int, look_key: str) -> str:
    look = LOOKS[look_key]
    T = max(0.1, shot["out_len"])
    ramp = shot.get("ramp")
    if ramp and abs(ramp[1] - ramp[0]) > 1e-3:
        a, b = ramp
        k = (b - a) / (2 * T)  # source time s(t) = a*t + k*t^2, inverted to output time
        parts = ["setpts=PTS-STARTPTS", f"setpts='(-({a})+sqrt(max(0,{a * a}+4*({k})*T)))/(2*({k}))/TB'", f"fps={fps}"]
    else:
        parts = [f"setpts=(PTS-STARTPTS)/{shot['speed']}", f"fps={fps}"]
    if rot := rotate_filter(shot.get("rotate_fix", 0)):
        parts.insert(0, rot.rstrip(","))
    crop = shot["crop"]
    a_out = W / H
    geo = []
    if crop["mode"] == "crop" and crop["axis"] == "x":
        p0, p1 = crop["pos0"], crop["pos1"]
        xexpr = f"(iw-ow)*({p0}+({p1 - p0})*(0.5-0.5*cos(PI*min(t/{T},1))))"
        geo.append(f"crop=w='floor(ih*{a_out:.6f}/2)*2':h=ih:x='{xexpr}':y=0")
        geo.append(f"scale={W}:{H}:flags=lanczos")
    elif crop["mode"] == "crop":
        p0 = crop["pos0"]
        geo.append(f"crop=w=iw:h='floor(iw/{a_out:.6f}/2)*2':x=0:y='(ih-oh)*{p0}'")
        geo.append(f"scale={W}:{H}:flags=lanczos")
    elif crop["mode"] == "blurfill":
        geo.append(
            f"split[bgs][fgs];[bgs]scale={W // 4}:{H // 4}:force_original_aspect_ratio=increase,crop={W // 4}:{H // 4},"
            f"gblur=sigma=12,scale={W}:{H},eq=brightness=-0.06:saturation=0.8[bg];"
            f"[fgs]scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
        )
    else:
        geo.append(f"scale={W}:{H}:flags=lanczos")
    parts += geo
    if shot.get("punch"):
        z = f"(1+{shot['punch']}*pow(max(0,1-t/0.35),2))"
        parts.append(f"scale=w='trunc(iw*{z}/2)*2':h='trunc(ih*{z}/2)*2':eval=frame,crop={W}:{H}")
    lut = lut_path(shot["profile"], look_key, shot.get("ev", 0.0))
    parts.append(f"scale={W}:{H}:{color_in(clip.get('color_space', ''), clip.get('color_range', 'tv'))},format=gbrp16le")
    parts.append(f"lut3d=file='{_esc(lut)}':interp=tetrahedral")
    if look.glow:
        sigma = max(4, W // 70)
        parts.append(f"split[g0][g1];[g1]gblur=sigma={sigma}[gb];[g0][gb]blend=all_mode=screen:all_opacity={look.glow}")
    if look.vignette:
        parts.append(f"vignette=angle={0.25 + look.vignette:.3f}")
    parts.append("scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,setsar=1")
    if shot.get("flash"):
        parts.append("fade=t=in:st=0:d=0.2:color=white")
    return ",".join(parts)


def render_shot(shot: dict, clip: dict, W: int, H: int, fps: int, look_key: str, out: Path, quality: str, keep_audio: bool):
    vf = shot_video_filter(shot, clip, W, H, fps, look_key)
    T = shot["out_len"]
    cmd = [FFMPEG, "-y", "-v", "error", "-hwaccel", "videotoolbox",
           "-ss", f"{shot['start']:.3f}", "-t", f"{shot['src_dur'] + 0.2:.3f}", "-i", clip["path"],
           "-f", "lavfi", "-t", f"{T:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
    use_src_audio = keep_audio and clip.get("has_audio") and abs(shot["speed"] - 1.0) < 1e-3 and not shot.get("ramp")
    if use_src_audio:
        af = f"[0:a]aresample=48000,aformat=channel_layouts=stereo,afade=t=in:d=0.08,afade=t=out:st={max(0, T - 0.12):.3f}:d=0.12,apad[a]"
    else:
        af = "[1:a]anull[a]"
    fc = f"[0:v]{vf}[v];{af}"
    preset, crf = ("ultrafast", "24") if quality == "preview" else ("fast", "15")
    cmd += ["-filter_complex", fc, "-map", "[v]", "-map", "[a]", "-t", f"{T:.3f}",
            "-r", str(fps), "-c:v", "libx264", "-preset", preset, "-crf", crf, "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", str(out)]
    try:
        run(cmd)
    except RuntimeError:
        cmd.pop(cmd.index("-hwaccel") + 1)
        cmd.remove("-hwaccel")
        run(cmd)


def title_png(text: str, sub: str, W: int, H: int, out: Path):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    size = int(W * 0.13) if W < H else int(H * 0.14)

    def font(sz, idx):
        for path in (FONT_TITLE, FONT_FALLBACK):
            try:
                return ImageFont.truetype(path, sz, index=idx)
            except OSError:
                continue
        return ImageFont.load_default()

    f_main, f_sub = font(size, 2), font(int(size * 0.36), 0)
    main = " ".join(text.upper())
    d = ImageDraw.Draw(img)
    mb = d.textbbox((0, 0), main, font=f_main)
    sb = d.textbbox((0, 0), sub.upper(), font=f_sub) if sub else (0, 0, 0, 0)
    cy = H * (0.42 if W < H else 0.5)
    gap = size * 0.42
    mx, my = (W - (mb[2] - mb[0])) / 2 - mb[0], cy - gap / 2 - mb[3]
    sx, sy = (W - (sb[2] - sb[0])) / 2 - sb[0], cy + gap / 2 - sb[1]
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.text((mx, my), main, font=f_main, fill=(0, 0, 0, 150))
    if sub:
        sd.text((sx, sy), sub.upper(), font=f_sub, fill=(0, 0, 0, 150))
    shadow = shadow.filter(ImageFilter.GaussianBlur(size * 0.12))
    img = Image.alpha_composite(img, shadow)
    d = ImageDraw.Draw(img)
    d.text((mx, my), main, font=f_main, fill=(255, 255, 255, 240))
    if sub:
        d.text((sx, sy), sub.upper(), font=f_sub, fill=(255, 255, 255, 210))
        lw = (sb[2] - sb[0]) * 0.5
        d.line([(W / 2 - lw / 2, cy), (W / 2 + lw / 2, cy)], fill=(255, 255, 255, 160), width=max(1, W // 600))
    img.save(out)


def assemble(plan: dict, seg_paths: list[Path], W: int, H: int, out: Path, quality: str, music: dict | None, workdir: Path):
    look = LOOKS[plan["look"]]
    shots = plan["shots"]
    fps = plan["fps"]
    total = plan["duration"]
    inputs: list[str] = []
    for p in seg_paths:
        inputs += ["-i", str(p)]
    fc: list[str] = []
    for i in range(len(seg_paths)):
        fc.append(f"[{i}:v]setpts=PTS-STARTPTS,fps={fps},settb=AVTB[v{i}]")
        fc.append(f"[{i}:a]asetpts=PTS-STARTPTS[a{i}]")
    vacc, aacc, acc_len = "v0", "a0", shots[0]["out_len"]
    for k in range(1, len(seg_paths)):
        tr, d = shots[k]["transition_in"]["type"], shots[k]["transition_in"]["dur"]
        vo, ao = f"vx{k}", f"ax{k}"
        if tr == "cut" or d <= 0:
            fc.append(f"[{vacc}][{aacc}][v{k}][a{k}]concat=n=2:v=1:a=1[{vo}][{ao}]")
            acc_len += shots[k]["out_len"]
        else:
            off = max(0.0, acc_len - d)
            fc.append(f"[{vacc}][v{k}]xfade=transition={tr}:duration={d:.3f}:offset={off:.3f}[{vo}]")
            fc.append(f"[{aacc}][a{k}]acrossfade=d={d:.3f}:c1=tri:c2=tri[{ao}]")
            acc_len += shots[k]["out_len"] - d
        vacc, aacc = vo, ao

    vchain = [f"trim=duration={total:.3f}", "setpts=PTS-STARTPTS"]
    if look.letterbox and plan["settings"].get("letterbox", True) and W / H >= 1.5:
        bar = _even((H - W / 2.39) / 2)
        vchain.append(f"drawbox=x=0:y=0:w=iw:h={bar}:color=black:t=fill,drawbox=x=0:y=ih-{bar}:w=iw:h={bar}:color=black:t=fill")
    if look.grain:
        vchain.append(f"noise=alls={look.grain}:allf=t")
    if look.fps <= 25:
        vchain.append("fade=t=in:st=0:d=0.4")
    vchain.append(f"fade=t=out:st={max(0, total - 0.8):.3f}:d=0.8")
    fc.append(f"[{vacc}]{','.join(vchain)}[vbase]")
    vlast = "vbase"

    title = (plan["settings"].get("title") or "").strip()
    n_in = len(seg_paths)
    if title:
        png = workdir / "title.png"
        title_png(title, plan["settings"].get("subtitle", ""), W, H, png)
        inputs += ["-loop", "1", "-t", "3.6", "-i", str(png)]
        fc.append(f"[{n_in}:v]format=rgba,fade=t=in:st=0.3:d=0.7:alpha=1,fade=t=out:st=2.7:d=0.8:alpha=1[ttl]")
        fc.append(f"[{vlast}][ttl]overlay=0:0:eof_action=pass:format=auto[vt]")
        vlast = "vt"
        n_in += 1

    ambient = float(plan["settings"].get("ambient", 0.25 if music else 1.0))
    fc.append(f"[{aacc}]atrim=duration={total:.3f},asetpts=PTS-STARTPTS,volume={ambient:.3f}[amb]")
    if music:
        inputs += ["-i", music["path"]]
        mvol = float(plan["settings"].get("music_volume", 0.9))
        fc.append(f"[{n_in}:a]atrim=start={music['offset']:.3f}:duration={total:.3f},asetpts=PTS-STARTPTS,"
                  f"aresample=48000,aformat=channel_layouts=stereo,volume={mvol:.3f},"
                  f"afade=t=out:st={max(0, total - 1.8):.3f}:d=1.8[mus]")
        fc.append("[amb][mus]amix=inputs=2:normalize=0:duration=first[amix]")
        alast = "amix"
    else:
        fc.append(f"[amb]afade=t=out:st={max(0, total - 1.0):.3f}:d=1.0[afin]")
        alast = "afin"
    fc.append(f"[{alast}]alimiter=limit=0.95[aout]")

    if quality == "preview":
        venc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "25"]
    else:
        venc = ["-c:v", "libx264", "-preset", "medium", "-crf", "18" if W * H <= 1920 * 1920 else "20", "-profile:v", "high"]
    cmd = [FFMPEG, "-y", "-v", "error", *inputs, "-filter_complex", ";".join(fc),
           "-map", f"[{vlast}]", "-map", "[aout]", "-t", f"{total:.3f}", "-r", str(fps),
           *venc, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", str(out)]
    (workdir / "assemble_cmd.json").write_text(json.dumps(cmd, indent=1))
    run(cmd)


def render_plan(plan: dict, clips: dict[str, dict], out: Path, workdir: Path, quality: str = "preview",
                music: dict | None = None, progress: Callable[[float, str], None] | None = None) -> dict:
    scale = 0.5 if quality == "preview" else 1.0
    W, H = _even(plan["width"] * scale), _even(plan["height"] * scale)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    keep_audio = plan["settings"].get("ambient", 1) > 0 or not music
    segs = [workdir / f"seg_{i:03d}.mp4" for i in range(len(plan["shots"]))]
    done = 0

    def work(i):
        shot = plan["shots"][i]
        render_shot(shot, clips[shot["clip_id"]], W, H, plan["fps"], plan["look"], segs[i], quality, keep_audio)
        return i

    with ThreadPoolExecutor(max_workers=3) as ex:
        for _ in ex.map(control.bind(work), range(len(segs))):
            done += 1
            if progress:
                progress(0.85 * done / len(segs), f"Rendered shot {done}/{len(segs)}")
    if progress:
        progress(0.88, "Assembling cut, transitions and music")
    assemble(plan, segs, W, H, out, quality, music, workdir)
    thumbs = []
    if quality == "preview":
        for i, seg in enumerate(segs):
            th = out.parent / f"{out.stem}_shot{i:02d}.jpg"
            mid = plan["shots"][i]["out_len"] / 2
            control.run([FFMPEG, "-y", "-v", "error", "-ss", f"{mid:.2f}", "-i", str(seg), "-frames:v", "1",
                         "-vf", "scale=240:-2", "-q:v", "4", str(th)])
            thumbs.append(th.name)
    shutil.rmtree(workdir, ignore_errors=True)
    if progress:
        progress(1.0, "Done")
    return {"file": out.name, "width": W, "height": H, "thumbs": thumbs}


def clip_frames(clip: dict, out_sprite: Path, out_thumb: Path, sheet: Path | None = None, n: int = 4):
    """Graded (log-normalised) frames for the library: a hover-scrub sprite and a thumbnail."""
    lut = lut_path(clip["profile"], None, 0.0)
    w = 320
    h = _even(w * clip["height"] / clip["width"])
    frames = []
    for k in range(n):
        t = clip["duration"] * (0.15 + 0.7 * k / max(1, n - 1))
        tmp = out_sprite.parent / f".{clip['id']}_{k}.jpg"
        vf = f"{rotate_filter(clip.get('rotate_fix', 0))}scale={w}:{h}:{color_in(clip.get('color_space', ''), clip.get('color_range', 'tv'))},format=gbrp16le,lut3d=file='{_esc(lut)}',format=yuvj420p"
        control.run([FFMPEG, "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", clip["path"], "-frames:v", "1", "-vf", vf, "-q:v", "4", str(tmp)])
        if tmp.exists():
            frames.append(Image.open(tmp).convert("RGB"))
            tmp.unlink()
    if not frames:
        return False
    sprite = Image.new("RGB", (w * len(frames), h))
    for k, f in enumerate(frames):
        sprite.paste(f.resize((w, h)), (k * w, 0))
    sprite.save(out_sprite, quality=82)
    frames[min(1, len(frames) - 1)].save(out_thumb, quality=85)
    if sheet is not None:
        cols = 2
        rows = (len(frames) + 1) // 2
        s = Image.new("RGB", (w * cols, h * rows))
        for k, f in enumerate(frames):
            s.paste(f.resize((w, h)), ((k % cols) * w, (k // cols) * h))
        s.save(sheet, quality=80)
    return True
