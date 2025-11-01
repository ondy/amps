#!/usr/bin/env python3
import argparse
import gzip
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

# Erwartete Eingangsstruktur:
# <in_root>/<language>/amplitude-<language>-<deck>-<songid>.csv
# Spalten: time_s, rms, peak  (Gleitkomma, time_s monoton)

INFILE_RE = re.compile(r"^amplitude-([a-z0-9\-]+)-([a-z0-9\-]+)-(.+)\.csv$", re.IGNORECASE)

def find_input_files(in_root: Path):
    for lang_dir in sorted(in_root.iterdir()):
        if not lang_dir.is_dir():
            continue
        for csvf in sorted(lang_dir.glob("amplitude-*-*-*.csv")):
            m = INFILE_RE.match(csvf.name)
            if not m:
                continue
            language, deck, songid = m.groups()
            yield language.lower(), deck.lower(), songid, csvf

def bin_and_quantize(df: pd.DataFrame, interval_ms: int):
    if df.empty:
        return pd.DataFrame(columns=["t_ms", "rms8", "peak8"])

    # Sicherstellen, dass erwartete Spalten existieren
    need = {"time_s", "rms", "peak"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"CSV hat unerwartetes Format: {df.columns.tolist()}")

    time_s = df["time_s"].to_numpy(dtype=float)
    rms = df["rms"].to_numpy(dtype=float)
    peak = df["peak"].to_numpy(dtype=float)

    # Negative oder NaN wegräumen
    mask = np.isfinite(time_s) & np.isfinite(rms) & np.isfinite(peak)
    time_s = time_s[mask]
    rms = rms[mask]
    peak = peak[mask]

    if time_s.size == 0:
        return pd.DataFrame(columns=["t_ms", "rms8", "peak8"])

    # Zeit-Binning
    interval_s = interval_ms / 1000.0
    t0 = time_s[0]
    bins = np.floor((time_s - t0) / interval_s).astype(np.int64)

    # Für jede Bin: t_ms = linker Rand der Bin, rms = mean, peak = max
    # Wir aggregieren effizient mit pandas
    agg = (
        pd.DataFrame({
            "bin": bins,
            "t_ms": ((bins * interval_s + t0) * 1000.0).astype(np.int64),
            "rms": rms,
            "peak": peak
        })
        .groupby("bin", as_index=False)
        .agg({
            "t_ms": "first",
            "rms": "mean",
            "peak": "max"
        })
        .sort_values("t_ms")
        .reset_index(drop=True)
    )

    # Auf [0,1] clippen (rms/peak sind aus der Analyse bereits normiert, aber sicher ist sicher)
    rms_c = np.clip(agg["rms"].to_numpy(), 0.0, 1.0)
    peak_c = np.clip(agg["peak"].to_numpy(), 0.0, 1.0)

    # 8-Bit Quantisierung
    rms8 = np.rint(rms_c * 255.0).astype(np.uint8)
    peak8 = np.rint(peak_c * 255.0).astype(np.uint8)

    out = pd.DataFrame({
        "t_ms": agg["t_ms"].astype(np.int64),
        "rms8": rms8,
        "peak8": peak8
    })
    return out

def write_csv_gz(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8", newline="") as gz:
        df.to_csv(gz, index=False)

def main():
    ap = argparse.ArgumentParser(description="Komprimiere Hitster-Amplituden: Downsampling + 8-bit Quantisierung + GZip.")
    ap.add_argument("--in", dest="in_root", type=str, default="amplitudes",
                    help="Eingangs-Wurzelordner (mit Unterordnern je Sprache).")
    ap.add_argument("--out", dest="out_root", type=str, default="amplitudes_small",
                    help="Ziel-Wurzelordner (Sprach-Unterordner werden gespiegelt).")
    ap.add_argument("--interval-ms", type=int, default=100,
                    help="Zeitauflösung der Bins in Millisekunden (Standard 100ms = 10 Hz).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Vorhandene Ziel-Dateien überschreiben (sonst überspringen).")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)

    if not in_root.exists():
        print(f"Eingangsordner nicht gefunden: {in_root}", file=sys.stderr)
        sys.exit(1)

    total = 0
    skipped = 0
    written = 0
    errors = 0

    for language, deck, songid, csvf in find_input_files(in_root):
        total += 1

        out_dir = out_root / language
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"amplitude8-{language}-{deck}-{songid}.csv.gz"
        out_path = out_dir / out_name

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            df = pd.read_csv(csvf)
            compact = bin_and_quantize(df, interval_ms=args.interval_ms)
            write_csv_gz(compact, out_path)
            written += 1
        except Exception as e:
            errors += 1
            print(f"[Fehler] {csvf} -> {e}", file=sys.stderr)

    print(f"\nFertig.")
    print(f"  Eingelesen: {total}")
    print(f"  Geschrieben: {written}")
    print(f"  Übersprungen (vorhanden): {skipped}")
    print(f"  Fehler: {errors}")
    print(f"\nAusgabe unter: {out_root}/<language>/amplitude8-<language>-<deck>-<songid>.csv.gz")

if __name__ == "__main__":
    main()
