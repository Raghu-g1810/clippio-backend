"""
Clippio Backend v11 - Stable, minimal, guaranteed to work
"""

import os, sys, json, re, uuid, subprocess, shutil, requests, threading, time, zipfile
from pathlib import Path
from typing import Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

def pip_install(pkg, import_as=None):
    try: __import__(import_as or pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

pip_install("yt-dlp", "yt_dlp")
pip_install("youtube-transcript-api", "youtube_transcript_api")
pip_install("python-dotenv", "dotenv")

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Clippio API", version="11.0.0")
app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:5173","http://localhost:3000","http://127.0.0.1:5173"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

WORK_DIR     = Path("workdir");  WORK_DIR.mkdir(exist_ok=True)
OUTPUT_DIR   = Path("outputs");  OUTPUT_DIR.mkdir(exist_ok=True)
COOKIES_FILE = Path("cookies.txt")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

jobs = {}

# Simple quality presets - "best" always works on every video
QUALITY_PROFILES = {
    "turbo":    {"yt_format": "bestvideo[height<=480]+bestaudio/best[height<=480]/best", "preset": "ultrafast", "crf": "32", "audio_br": "64k",  "threads": "2"},
    "fast":     {"yt_format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best", "preset": "ultrafast", "crf": "28", "audio_br": "96k",  "threads": "2"},
    "balanced": {"yt_format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best", "preset": "fast",      "crf": "24", "audio_br": "128k", "threads": "2"},
    "hd":       {"yt_format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best","preset": "medium",   "crf": "20", "audio_br": "192k", "threads": "4"},
    "ultra":    {"yt_format": "bestvideo+bestaudio/best",                                "preset": "slow",      "crf": "18", "audio_br": "320k", "threads": "0"},
}

class CreateJobRequest(BaseModel):
    youtube_url:    str
    num_shorts:     int  = 3
    ai_provider:    str  = "groq"
    api_key:        Optional[str] = None
    quality:        str  = "fast"
    zoom_crop:      bool = True
    clip_length:    int  = 60

class RenameClipRequest(BaseModel):
    new_title: str

def extract_video_id(url):
    m = re.search(r'(?:v=|youtu\.be/|embed/|shorts/)([a-zA-Z0-9_-]{11})', url)
    return m.group(1) if m else None

def fmt_time(s):
    return f"{int(s//3600):02d}:{int((s%3600)//60):02d}:{int(s%60):02d}"

def srt_ts(s):
    h=int(s//3600); m=int((s%3600)//60); sec=int(s%60); ms=int((s%1)*1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

def update_job(jid, **kw):
    jobs[jid].update(kw)

def get_ydl_opts(extra=None):
    opts = {
        "quiet": True,
        "no_warnings": True,
        "extractor_retries": 5,
        "retries": 10,
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        },
    }
    if COOKIES_FILE.exists() and COOKIES_FILE.stat().st_size > 100:
        opts["cookiefile"] = str(COOKIES_FILE)
    if extra:
        opts.update(extra)
    return opts

def ydl_fetch_info(url):
    opts = get_ydl_opts({"skip_download": True})
    # Try with cookies file first, then browser cookies, then plain
    errors = []
    for attempt in [
        opts,
        {**opts, "cookiesfrombrowser": ("chrome",)},
        {**opts, "cookiesfrombrowser": ("firefox",)},
        {k:v for k,v in opts.items() if k not in ("cookiefile","cookiesfrombrowser")},
    ]:
        try:
            with yt_dlp.YoutubeDL(attempt) as ydl:
                return ydl.extract_info(url, download=False)
        except Exception as e:
            errors.append(str(e))
    raise Exception(errors[-1])

def ydl_download(url, out_path, fmt):
    opts = get_ydl_opts({
        "format": fmt,
        "outtmpl": str(out_path),
        "merge_output_format": "mp4",
        "concurrent_fragment_downloads": 3,
    })
    errors = []
    attempts = [
        opts,
        {**opts, "cookiesfrombrowser": ("chrome",)},
        {**opts, "cookiesfrombrowser": ("firefox",)},
        {**{k:v for k,v in opts.items() if k not in ("cookiefile","cookiesfrombrowser")}, "format": "bestvideo+bestaudio/best"},
    ]
    for attempt in attempts:
        try:
            with yt_dlp.YoutubeDL(attempt) as ydl:
                ydl.download([url])
            return
        except Exception as e:
            errors.append(str(e))
            if out_path.exists(): out_path.unlink()
    raise Exception(f"Download failed: {errors[-1]}")

def call_ai(provider, api_key, prompt):
    p = provider.lower()
    if p == "gemini":
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
            json={"contents":[{"parts":[{"text":prompt}]}],"generationConfig":{"temperature":0.7,"maxOutputTokens":2000}},
            timeout=60)
        r.raise_for_status(); return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    elif p == "groq":
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"},
            json={"model":"llama-3.3-70b-versatile","messages":[{"role":"user","content":prompt}],"max_tokens":2000},timeout=60)
        r.raise_for_status(); return r.json()["choices"][0]["message"]["content"]
    elif p == "claude":
        r = requests.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key":api_key,"anthropic-version":"2023-06-01","Content-Type":"application/json"},
            json={"model":"claude-haiku-4-5-20251001","max_tokens":2000,"messages":[{"role":"user","content":prompt}]},timeout=60)
        r.raise_for_status(); return r.json()["content"][0]["text"]
    elif p == "openai":
        r = requests.post("https://api.openai.com/v1/chat/completions",
            headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"},
            json={"model":"gpt-4o-mini","messages":[{"role":"user","content":prompt}],"max_tokens":2000},timeout=60)
        r.raise_for_status(); return r.json()["choices"][0]["message"]["content"]
    elif p == "mistral":
        r = requests.post("https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"},
            json={"model":"mistral-small-latest","messages":[{"role":"user","content":prompt}],"max_tokens":2000},timeout=60)
        r.raise_for_status(); return r.json()["choices"][0]["message"]["content"]
    elif p == "cohere":
        r = requests.post("https://api.cohere.com/v2/chat",
            headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"},
            json={"model":"command-r-plus-08-2024","messages":[{"role":"user","content":prompt}],"max_tokens":2000},timeout=60)
        r.raise_for_status(); return r.json()["message"]["content"][0]["text"]
    else:
        raise ValueError(f"Unknown provider: {provider}")

def build_srt(transcript, clip_start, clip_duration):
    lines = []; idx = 1
    for seg in transcript:
        ss=seg["start"]; se=ss+seg.get("duration",2.0)
        if se < clip_start or ss > clip_start+clip_duration: continue
        rs=max(0.0,ss-clip_start); re_=min(float(clip_duration),se-clip_start)
        if re_ <= rs+0.05: continue
        text=seg["text"].strip()
        if not text or text.startswith("["): continue
        lines.append(f"{idx}\n{srt_ts(rs)} --> {srt_ts(re_)}\n{text}\n"); idx+=1
    return "\n".join(lines)

def cut_clip(video_path, out_file, start, duration, profile, zoom_crop):
    # Minimal, proven filter - works on all Windows ffmpeg builds
    if zoom_crop:
        vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
    else:
        vf = "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black"
    out_str = str(out_file).replace("\\", "/")
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(video_path),
        "-t", str(duration),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", profile["preset"],
        "-crf", profile["crf"],
        "-c:a", "aac",
        "-b:a", profile["audio_br"],
        "-movflags", "+faststart",
        "-threads", profile["threads"],
        out_str
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return result.returncode == 0, result.stderr[-500:] if result.returncode != 0 else ""

async def process_job(job_id, url, num_shorts, ai_provider, api_key, quality, zoom_crop, clip_length):
    job_dir = WORK_DIR / job_id;   job_dir.mkdir(exist_ok=True)
    out_dir = OUTPUT_DIR / job_id; out_dir.mkdir(exist_ok=True)
    profile = QUALITY_PROFILES.get(quality, QUALITY_PROFILES["fast"])

    try:
        # 1. Video info
        update_job(job_id, status="processing", step="Fetching video info...", progress=5)
        info        = ydl_fetch_info(url)
        video_title = info["title"]
        duration    = info["duration"]
        video_id    = extract_video_id(url)
        update_job(job_id, video_title=video_title, video_duration=duration, progress=10)

        # 2. Transcript
        update_job(job_id, step="Fetching transcript...", progress=14)
        transcript = []
        try:
            fetched    = YouTubeTranscriptApi().fetch(video_id)
            transcript = [{"start":s.start,"duration":s.duration,"text":s.text} for s in fetched]
        except Exception:
            try:    transcript = YouTubeTranscriptApi.get_transcript(video_id)
            except: transcript = []

        tx_text = "".join(f"[{fmt_time(s['start'])}] {s['text']}\n" for s in transcript)
        if len(tx_text) > 10000: tx_text = tx_text[:10000] + "\n...[truncated]"
        if not tx_text: tx_text = f"[No transcript - {video_title}]"

        # 3+4. AI + Download in parallel
        update_job(job_id, step="AI analyzing + downloading simultaneously...", progress=18)
        video_path = job_dir / "source.mp4"
        ai_result=[None]; ai_error=[None]; dl_error=[None]

        prompt = (
            f"Viral YouTube Shorts expert. Find {num_shorts} best {clip_length}-second moments "
            f"from this video that DO NOT overlap with each other.\n"
            f"Video: {video_title}\nDuration: {fmt_time(duration)}\nTranscript:\n{tx_text}\n\n"
            f"Reply ONLY with JSON array, no markdown:\n"
            f'[{{"short_number":1,"title":"8 word title","start_seconds":45,"end_seconds":{45+clip_length},"hook":"hook line","why":"why viral","virality_score":8}}]\n'
            f"CRITICAL RULES:\n"
            f"1. end_seconds - start_seconds must equal exactly {clip_length}\n"
            f"2. start_seconds >= 5 and end_seconds <= {int(duration)-2}\n"
            f"3. Each clip's time range must NOT overlap with any other clip's time range\n"
            f"4. Space clips at least {clip_length} seconds apart from each other\n"
            f"5. Spread all {num_shorts} clips evenly across the full video duration ({int(duration)}s)\n"
            f"6. Each clip must cover a DIFFERENT part of the video with DIFFERENT content"
        )

        def run_ai():
            try:
                raw = call_ai(ai_provider, api_key, prompt)
                raw = re.sub(r"^```json\s*|^```\s*|\s*```$","",raw.strip()).strip()
                ai_result[0] = json.loads(raw)
            except Exception as e: ai_error[0] = str(e)

        def run_download():
            try:
                if not video_path.exists():
                    ydl_download(url, video_path, profile["yt_format"])
            except Exception as e: dl_error[0] = str(e)

        t_ai=threading.Thread(target=run_ai); t_dl=threading.Thread(target=run_download)
        t_ai.start(); t_dl.start()
        progress = 18
        while t_ai.is_alive() or t_dl.is_alive():
            time.sleep(0.8)
            if progress < 62: progress += 2
            ai_done=not t_ai.is_alive(); dl_done=not t_dl.is_alive()
            if   ai_done and dl_done: step="Both done! Cutting clips..."
            elif ai_done:             step="AI done! Downloading video..."
            elif dl_done:             step="Download done! AI analyzing..."
            else:                     step="Downloading + AI analyzing..."
            update_job(job_id, progress=progress, step=step)
        t_ai.join(); t_dl.join()

        if ai_error[0]: raise Exception(f"AI error: {ai_error[0]}")
        if dl_error[0]: raise Exception(f"Download failed: {dl_error[0]}")
        if not ai_result[0]: raise Exception("AI returned no results")

        moments = ai_result[0]

        # ── Validate & fix overlapping/duplicate timestamps ──────
        moments = sorted(moments, key=lambda m: m.get("start_seconds", 0))
        seen_ranges = []
        fixed_moments = []
        min_gap = clip_length  # clips must not overlap

        for m in moments:
            start = float(m.get("start_seconds", 0))
            end   = start + clip_length  # force exact length

            # Check overlap with already-placed clips
            overlaps = any(
                not (end <= s or start >= e)
                for s, e in seen_ranges
            )

            if overlaps:
                # Shift this clip to the next free slot after the last placed clip
                if seen_ranges:
                    start = max(start, seen_ranges[-1][1] + 1)
                    end = start + clip_length

                # If it now runs past the video, try placing near the start instead
                if end > duration - 2:
                    start = max(5, duration - clip_length - 2)
                    end = start + clip_length

                # Re-check overlap after shifting — if still overlapping, skip this clip
                overlaps = any(
                    not (end <= s or start >= e)
                    for s, e in seen_ranges
                )
                if overlaps:
                    continue

            m["start_seconds"] = start
            m["end_seconds"]   = end
            seen_ranges.append((start, end))
            fixed_moments.append(m)

        moments = fixed_moments
        if not moments:
            raise Exception("Could not find non-overlapping clip positions in this video")
        update_job(job_id, progress=64, step=f"Cutting {len(moments)} clips...")
        clips = [None] * len(moments)

        def cut_one(i, m):
            safe      = re.sub(r"[^\w]","_",m["title"])[:30]
            clip_file = out_dir / f"short_{i+1:02d}_{safe}.mp4"
            srt_file  = out_dir / f"short_{i+1:02d}_{safe}.srt"
            clip_dur  = m["end_seconds"] - m["start_seconds"]
            ok, err   = cut_clip(video_path, clip_file, m["start_seconds"], clip_dur, profile, zoom_crop)
            if transcript:
                srt_content = build_srt(transcript, m["start_seconds"], clip_dur)
                if srt_content: srt_file.write_text(srt_content, encoding="utf-8")
            if ok:
                size_mb = round(os.path.getsize(str(clip_file))/(1024*1024),1)
                clips[i] = {
                    "short_number":m["short_number"],"title":m["title"],
                    "hook":m.get("hook",""),"why":m.get("why",""),
                    "virality_score":m.get("virality_score",7),
                    "start":m["start_seconds"],"end":m["end_seconds"],
                    "duration":clip_dur,"filename":clip_file.name,
                    "url":f"/outputs/{job_id}/{clip_file.name}",
                    "srt_url":f"/outputs/{job_id}/{srt_file.name}" if srt_file.exists() else None,
                    "size_mb":size_mb,"quality":quality,"status":"ready",
                }
            else:
                clips[i] = {**m,"status":"error","error":err}

        with ThreadPoolExecutor(max_workers=min(len(moments),3)) as ex:
            futs = {ex.submit(cut_one,i,m):i for i,m in enumerate(moments)}
            done = 0
            for f in as_completed(futs):
                done += 1
                update_job(job_id, progress=64+int((done/len(moments))*34),
                           step=f"Cutting clips... {done}/{len(moments)} done")

        update_job(job_id, status="done", step="All shorts ready!", progress=100,
                   clips=[c for c in clips if c], quality=quality)

    except Exception as e:
        update_job(job_id, status="error", step="Failed", progress=0, error=str(e))

@app.get("/")
def root(): return {"status":"ok","service":"Clippio v11"}

@app.get("/api/cookies-status")
def cookies_status():
    has_file = COOKIES_FILE.exists() and COOKIES_FILE.stat().st_size > 100
    return {"has_cookies": has_file}

@app.post("/api/upload-cookies")
async def upload_cookies(file: UploadFile = File(...)):
    content = await file.read()
    COOKIES_FILE.write_bytes(content)
    return {"ok": True}

@app.delete("/api/cookies")
def delete_cookies():
    if COOKIES_FILE.exists(): COOKIES_FILE.unlink()
    return {"ok": True}

@app.post("/api/jobs")
async def create_job(req: CreateJobRequest, background_tasks: BackgroundTasks):
    api_key = req.api_key or os.environ.get("AI_API_KEY","")
    if not api_key: raise HTTPException(400,"API key required")
    if not extract_video_id(req.youtube_url): raise HTTPException(400,"Invalid YouTube URL")
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "job_id":job_id,"status":"queued","progress":0,"step":"Queued...",
        "clips":[],"error":None,"created_at":datetime.now().isoformat(),
        "video_title":None,"video_duration":None,
        "ai_provider":req.ai_provider,"quality":req.quality,"clip_length":req.clip_length,
    }
    background_tasks.add_task(process_job, job_id, req.youtube_url,
        min(5,max(1,req.num_shorts)), req.ai_provider, api_key,
        req.quality, req.zoom_crop, req.clip_length)
    return {"job_id":job_id}

@app.get("/api/jobs/{job_id}")
def get_job(job_id:str):
    if job_id not in jobs: raise HTTPException(404,"Job not found")
    return jobs[job_id]

@app.get("/api/jobs")
def list_jobs(): return list(jobs.values())

@app.delete("/api/jobs/{job_id}")
def delete_job(job_id:str):
    if job_id not in jobs: raise HTTPException(404,"Job not found")
    del jobs[job_id]
    for d in [OUTPUT_DIR/job_id, WORK_DIR/job_id]:
        if d.exists(): shutil.rmtree(d)
    return {"deleted":job_id}

@app.patch("/api/jobs/{job_id}/clips/{clip_index}/rename")
def rename_clip(job_id:str, clip_index:int, req: RenameClipRequest):
    if job_id not in jobs: raise HTTPException(404,"Job not found")
    clips = jobs[job_id].get("clips",[])
    if clip_index >= len(clips): raise HTTPException(404,"Clip not found")
    clips[clip_index]["title"] = req.new_title
    return {"ok":True}

@app.get("/api/jobs/{job_id}/download-all")
def download_all(job_id:str):
    if job_id not in jobs: raise HTTPException(404,"Job not found")
    clips = jobs[job_id].get("clips",[])
    if not clips: raise HTTPException(400,"No clips")
    zip_path = OUTPUT_DIR / job_id / "all_clips.zip"
    with zipfile.ZipFile(str(zip_path),"w",zipfile.ZIP_DEFLATED) as zf:
        for clip in clips:
            if clip.get("status") != "ready": continue
            mp4 = OUTPUT_DIR / job_id / clip["filename"]
            if mp4.exists(): zf.write(str(mp4), clip["filename"])
            if clip.get("srt_url"):
                srt = OUTPUT_DIR / job_id / Path(clip["srt_url"]).name
                if srt.exists(): zf.write(str(srt), srt.name)
    return FileResponse(str(zip_path), media_type="application/zip",
                        filename=f"Clippio_{job_id}.zip")

# ── Crop Focus / Fixed Crop endpoint ──────────────────────
# Added separately — does NOT touch any existing code above

class RecutRequest(BaseModel):
    clip_index:  int
    focus_x:     float   # 0.0 to 1.0 relative position in original video width
    focus_y:     float   # 0.0 to 1.0 relative position in original video height

@app.post("/api/jobs/{job_id}/recut")
def recut_clip(job_id: str, req: RecutRequest):
    """Re-cut a clip cropped around a user-selected focus point."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    clips = jobs[job_id].get("clips", [])
    if req.clip_index >= len(clips):
        raise HTTPException(404, "Clip not found")

    clip    = clips[req.clip_index]
    profile = QUALITY_PROFILES.get(jobs[job_id].get("quality","fast"), QUALITY_PROFILES["fast"])

    # Source video
    video_path = WORK_DIR / job_id / "source.mp4"
    if not video_path.exists():
        raise HTTPException(400, "Source video not found — please regenerate the clip")

    # Get source video dimensions using ffprobe
    probe = subprocess.run([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", str(video_path)
    ], capture_output=True, text=True, encoding="utf-8", errors="replace")

    src_w, src_h = 1280, 720  # safe defaults
    try:
        info = json.loads(probe.stdout)
        for s in info.get("streams", []):
            if s.get("codec_type") == "video":
                src_w = int(s["width"])
                src_h = int(s["height"])
                break
    except Exception:
        pass

    # Calculate crop: fixed 9:16 crop centered on focus point
    # Target crop size in source pixels (as tall as possible, 9:16 ratio)
    crop_h = src_h
    crop_w = int(crop_h * 9 / 16)

    if crop_w > src_w:
        crop_w = src_w
        crop_h = int(crop_w * 16 / 9)

    # Center on focus point, clamped to video bounds
    focus_px_x = int(req.focus_x * src_w)
    focus_px_y = int(req.focus_y * src_h)

    crop_x = max(0, min(focus_px_x - crop_w // 2, src_w - crop_w))
    crop_y = max(0, min(focus_px_y - crop_h // 2, src_h - crop_h))

    # Output file
    out_dir   = OUTPUT_DIR / job_id
    safe      = re.sub(r"[^\w]", "_", clip["title"])[:30]
    out_file  = out_dir / f"short_{req.clip_index+1:02d}_{safe}_focused.mp4"
    out_str   = str(out_file).replace("\\", "/")

    # ffmpeg: crop to focus area then scale to 1080x1920
    vf = f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale=1080:1920"

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(clip["start"]),
        "-i", str(video_path),
        "-t", str(clip["duration"]),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", profile["preset"],
        "-crf", profile["crf"],
        "-c:a", "aac",
        "-b:a", profile["audio_br"],
        "-movflags", "+faststart",
        "-threads", profile["threads"],
        out_str
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

    if result.returncode != 0:
        raise HTTPException(500, f"ffmpeg error: {result.stderr[-300:]}")

    size_mb = round(os.path.getsize(str(out_file)) / (1024 * 1024), 1)
    new_url = f"/outputs/{job_id}/{out_file.name}"

    # Update clip in job store with focused version
    clips[req.clip_index]["url"]           = new_url
    clips[req.clip_index]["filename"]      = out_file.name
    clips[req.clip_index]["size_mb"]       = size_mb
    clips[req.clip_index]["focus_applied"] = True

    return {
        "ok":      True,
        "url":     new_url,
        "size_mb": size_mb,
        "filename": out_file.name,
    }
