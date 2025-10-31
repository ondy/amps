#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from yt_dlp import YoutubeDL
import librosa

GITHUB_API = "https://api.github.com"
REPO_OWNER = "andygruber"
REPO_NAME = "songseeker-hitster-playlists"
DEFAULT_PATH = ""   # Repo-Root
DEFAULT_REF = "main"

HITSTER_REGEX = re.compile(r"^hitster-([a-z0-9\-]+)-([a-z0-9\-]+)\.csv$", re.IGNORECASE)

def gh_headers():
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

def list_repo_files(path: str, ref: str):
    url = f"{GITHUB_API}/repos/{REPO_OWNER}/{REPO_NAME}/contents/{path}"
    params = {"ref": ref} if ref else {}
    r = requests.get(url, headers=gh_headers(), params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def collect_hitster_csvs(path: str, ref: str):
    items = list_repo_files(path, ref)
    files = []
    for it in items:
        if it.get("type") == "file":
            name = it.get("name", "")
            m = HITSTER_REGEX.match(name)
            if m:
                language = m.group(1).lower()
                deck = m.group(2).lower()
                download_url = it.get("download_url")
                if download_url:
                    files.append({
                        "name": name,
                        "language": language,
                        "deck": deck,
                        "download_url": download_url
                    })
    return files

def download_audio(url: str, out_dir: Path) -> dict:
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "noplaylist": True,
        "ignoreerrors": False,
        "cookiefile": "youtube.com_cookies.txt",
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    return info

def to_wav_mono_22k(in_path: Path, out_path: Path):
    cmd = [
        "ffmpeg", "-y", "-i", str(in_path),
        "-ac", "1", "-ar", "22050",
        "-sample_fmt", "s16",
        str(out_path),
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

def frame_peaks(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if len(y) <= 0:
        return np.zeros(0, dtype=float)
    n_frames = max(1, 1 + (len(y) - frame_length) // hop_length) if len(y) >= frame_length else 1
    peaks = np.empty(n_frames, dtype=float)
    for i in range(n_frames):
        start = i * hop_length
        end = min(start + frame_length, len(y))
        if start >= len(y) or end <= start:
            peaks[i] = 0.0
        else:
            peaks[i] = float(np.max(np.abs(y[start:end])))
    return peaks

def analyze_wav(wav_path: Path, step_ms: float, window_ms: float):
    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    hop_length = max(1, int(sr * (step_ms / 1000.0)))
    frame_length = max(hop_length + 1, int(sr * (window_ms / 1000.0)))
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=False)[0]
    peaks = frame_peaks(y, frame_length=frame_length, hop_length=hop_length)
    n = min(len(rms), len(peaks))
    rms = rms[:n]
    peaks = peaks[:n]
    frames = np.arange(n)
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    return sr, frame_length, hop_length, times, rms, peaks

def safe_filename(name: str) -> str:
    return "".join(c for c in name if c not in '<>:"/\\|?*').strip()

def parse_csv_rows(csv_url: str):
    """
    Erwartet:
      - Spalte 1: songid
      - Spalte 5: YouTube-URL
    """
    try:
        df = pd.read_csv(csv_url, header=None, dtype=str, na_filter=False)
    except Exception:
        df = pd.read_csv(csv_url, header=0, dtype=str, na_filter=False)
    rows = []
    for _, row in df.iterrows():
        songid = (row.iloc[0] or "").strip() if len(row) >= 1 else ""
        yt = (row.iloc[4] or "").strip() if len(row) >= 5 else ""
        if songid and yt and yt.startswith("http"):
            rows.append((songid, yt))
    return rows

def main():
    ap = argparse.ArgumentParser(description="Amplitude-Zeitreihen (RMS+Peak) aus hitster-<lang>-<deck>.csv YouTube-Links exportieren.")
    ap.add_argument("--out", type=str, default="amplitude_exports", help="Ausgabeverzeichnis.")
    ap.add_argument("--step-ms", type=float, default=20.0, help="Hop/Step in Millisekunden (z.B. 20).")
    ap.add_argument("--window-ms", type=float, default=46.0, help="Fensterlänge in Millisekunden (z.B. 46).")
    ap.add_argument("--ref", type=str, default=DEFAULT_REF, help="Git-Ref/Branch (default: main).")
    ap.add_argument("--path", type=str, default=DEFAULT_PATH, help="Pfad im Repo (default: root).")
    ap.add_argument("--limit", type=int, default=0, help="Optional: Anzahl Tracks begrenzen (0 = alle).")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Suche Dateien im Repo {REPO_OWNER}/{REPO_NAME} (ref={args.ref}, path='{args.path or '/'}')...")
    files = collect_hitster_csvs(args.path, args.ref)
    if not files:
        print("Keine passenden hitster-<language>-<deck>.csv Dateien gefunden.", file=sys.stderr)
        sys.exit(1)

    catalog = []

    with tempfile.TemporaryDirectory() as tmpd:
        tmp_dir = Path(tmpd)

        for f in files:
            name = f["name"]
            language = f["language"]
            deck = f["deck"]
            csv_url = f["download_url"]

            # Sprache-Verzeichnis unter out_root anlegen, z.B. amplitude_exports/de
            lang_dir = out_root / language
            lang_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nLese {name} …")
            pairs = parse_csv_rows(csv_url)
            if not pairs:
                print(f"  -> keine (songid, url) Paare gefunden, überspringe.")
                continue

            if args.limit and len(pairs) > args.limit:
                pairs = pairs[:args.limit]

            for songid, yt_url in tqdm(pairs, desc=f"{name}", unit="track"):
                # Zielpfad (unter Sprache)
                out_name = f"amplitude-{language}-{deck}-{songid}.csv"
                out_path = lang_dir / safe_filename(out_name)

                # Wenn Datei existiert -> überspringen (keine Neu-Erzeugung)
                if out_path.exists():
                    tqdm.write(f"[Skip] Existiert bereits: {out_path}")
                    catalog.append({
                        "language": language,
                        "deck": deck,
                        "songid": songid,
                        "youtube_url": yt_url,
                        "video_id": None,
                        "title": None,
                        "uploader": None,
                        "duration_s": None,
                        "csv": str(out_path),
                        "status": "existing"
                    })
                    continue

                try:
                    # Download
                    info = download_audio(yt_url, tmp_dir)
                    vid = info.get("id") or "unknown"

                    # Eingangsdatei finden
                    orig = None
                    for p in tmp_dir.glob(f"{vid}.*"):
                        if p.suffix.lower() in (".m4a", ".webm", ".opus", ".mp3", ".mp4"):
                            orig = p
                            break
                    if orig is None:
                        cand = list(tmp_dir.glob(f"{vid}.*"))
                        if not cand:
                            raise RuntimeError("Download fehlgeschlagen (keine Datei gefunden).")
                        orig = cand[0]

                    # WAV konvertieren
                    wav_path = tmp_dir / f"{vid}.wav"
                    to_wav_mono_22k(orig, wav_path)

                    # Analyse
                    sr, frame_len, hop_len, times, rms, peaks = analyze_wav(
                        wav_path, step_ms=args.step_ms, window_ms=args.window_ms
                    )

                    # Schreiben
                    pd.DataFrame({"time_s": times, "rms": rms, "peak": peaks}).to_csv(out_path, index=False)

                    catalog.append({
                        "language": language,
                        "deck": deck,
                        "songid": songid,
                        "youtube_url": yt_url,
                        "video_id": vid,
                        "title": info.get("title"),
                        "uploader": info.get("uploader"),
                        "duration_s": info.get("duration"),
                        "csv": str(out_path),
                        "status": "created"
                    })

                except subprocess.CalledProcessError as e:
                    tqdm.write(f"[FFmpeg-Fehler] {yt_url}: {e}")
                except Exception as e:
                    tqdm.write(f"[Fehler] {yt_url}: {e}")

    # Übersicht oben im Root-Ordner
    if catalog:
        pd.DataFrame(catalog).to_csv(out_root / "overview.csv", index=False)
        meta = {
            "repo": f"{REPO_OWNER}/{REPO_NAME}",
            "ref": args.ref,
            "pattern": "hitster-<language>-<deck>.csv",
            "columns": ["time_s", "rms", "peak"],
            "step_ms": args.step_ms,
            "window_ms": args.window_ms,
            "out_structure": f"{out_root.name}/<language>/amplitude-<language>-<deck>-<songid>.csv"
        }
        with open(out_root / "run_meta.json", "w", encoding="utf-8") as mf:
            json.dump(meta, mf, ensure_ascii=False, indent=2)
        print(f"\nFertig. CSVs unter: {out_root}/<language>/ …  (Overview: {out_root/'overview.csv'})")
    else:
        print("\nNichts exportiert (siehe Meldungen oben).", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
