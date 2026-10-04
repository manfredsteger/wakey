#!/usr/bin/env python3
"""Holt automatisch bewertete Dobby-Fehlauslösungen als Trainings-Negatives.

Gegenstück zu tools/wakeword_harvest.py (läuft als Dienst in Proxmox-LXC 115):
  1. labels.jsonl aus LXC 115 lesen (ssh Proxmox-Host -> pct exec)
  2. WAV-Liste mit mtime von HA lesen (/share/assist_debug, ssh root@HA)
  3. jedem Label die WAV zuordnen, die innerhalb von MATCH_WINDOW_S nach
     run_start geschrieben wurde (Ordnername ist monotonic_ns, keine Run-ID)
  4. verdict=fp mit hörbarer Sprache -> output/hey_dobbi/negative_train/real_neg_fp_auto_*.wav
     real/unclear/stille fp -> output/hey_dobbi/harvest_review/<verdict>/ (nur zum Anhören)

Idempotent: bereits kopierte Dateien werden übersprungen.

Aufruf:  HARVEST_PVE_SSH=root@<pve> HARVEST_LXC_ID=<id> HARVEST_HA_SSH=root@<ha> \
         .venv/bin/python tools/harvest_sync.py [--dry-run]
"""
import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

# Zugänge kommen aus der Umgebung (private Adressen gehören nicht ins Repo)
PVE = os.environ.get("HARVEST_PVE_SSH", "")      # z. B. root@<proxmox-host>
LXC = os.environ.get("HARVEST_LXC_ID", "")       # Container mit wakeword-harvest
HA = os.environ.get("HARVEST_HA_SSH", "")        # z. B. root@<home-assistant>
LABELS = "/opt/mww-logger/harvest/labels.jsonl"
DEBUG_DIR = "/share/assist_debug"
MATCH_WINDOW_S = 40
SILENT_RMS_DB = -40.0

ROOT = Path(__file__).resolve().parent.parent
NEG_DIR = ROOT / "output/hey_dobbi/negative_train"
REVIEW_DIR = ROOT / "output/hey_dobbi/harvest_review"


def sh(*cmd: str) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120).stdout


def load_labels() -> list[dict]:
    out = sh("ssh", PVE, f"pct exec {LXC} -- cat {LABELS}")
    return [json.loads(l) for l in out.splitlines() if l.strip()]


def list_wavs() -> list[tuple[float, str]]:
    # BusyBox-find kennt kein -printf -> stat
    out = sh("ssh", HA, f"find {DEBUG_DIR} -name '01_stt-*.wav' -exec stat -c '%Y %n' {{}} +")
    rows = []
    for line in out.splitlines():
        ts, path = line.split(" ", 1)
        rows.append((float(ts), path))
    return sorted(rows)


def rms_db(path: Path) -> float:
    a, _ = sf.read(path)
    return float(20 * np.log10(np.sqrt(np.mean(np.square(a))) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not (PVE and LXC and HA):
        raise SystemExit("HARVEST_PVE_SSH, HARVEST_LXC_ID und HARVEST_HA_SSH setzen")

    labels, wavs = load_labels(), list_wavs()
    used: set[str] = set()
    stats: dict[str, int] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for lab in labels:
            t0 = datetime.fromisoformat(lab["run_start"]).timestamp()
            match = next((p for ts, p in wavs if t0 <= ts <= t0 + MATCH_WINDOW_S and p not in used), None)
            if match is None:
                stats["ohne_wav"] = stats.get("ohne_wav", 0) + 1
                continue
            used.add(match)
            stamp = datetime.fromtimestamp(t0).strftime("%Y%m%d_%H%M%S")
            local = Path(tmp) / f"{stamp}.wav"
            sh("scp", "-q", f"{HA}:{match}", str(local))
            verdict = lab["verdict"]
            if verdict == "fp" and rms_db(local) > SILENT_RMS_DB and lab["stt_text"].strip(" .…"):
                dest = NEG_DIR / f"real_neg_fp_auto_{stamp}.wav"
                key = "fp_negativ"
            else:
                sub = "fp_still" if verdict == "fp" else verdict
                dest = REVIEW_DIR / sub / f"{stamp}.wav"
                key = sub
            stats[key] = stats.get(key, 0) + 1
            if dest.exists() or a.dry_run:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(local.read_bytes())
            if key != "fp_negativ":
                dest.with_suffix(".txt").write_text(f"{lab['stt_text']}\n-> {lab['speech']}\n", encoding="utf-8")
    print(f"{len(labels)} Labels, {len(wavs)} WAVs auf HA:", stats)


if __name__ == "__main__":
    main()
