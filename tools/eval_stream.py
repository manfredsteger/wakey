#!/usr/bin/env python3
"""Streaming-Eval eines microWakeWord-Modells wie auf dem ESP32.

Bildet die ESPHome-Logik nach: Wahrscheinlichkeit je 10-ms-Schritt (Stride 3 →
alle 30 ms), gleitender Mittelwert über `sliding_window_size` Ausgaben, Auslösung
wenn Mittel > cutoff, danach Refraktärzeit (Fenster wird geleert).

  Erkennung (Recall): jede Datei in --pos einzeln, mit 1 s Stille davor/danach
  Fehlalarme:         jede --neg-Gruppe als ein Stream (Dateien hintereinander),
                      Ausgabe: Anzahl, FA/h und betroffene Dateien

Aufruf:
  .venv/bin/python tools/eval_stream.py --model output/hey_dobbi.tflite \
      --pos output/hey_dobbi/positive_test \
      --neg haushalt=output/hey_dobbi/harvest_holdout --neg cv=data/german_speech_eval
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "microWakeWord"))
from microwakeword.inference import Model  # noqa: E402

SR = 16000


def load(path: Path) -> np.ndarray:
    a, sr = sf.read(path, dtype="int16")
    if a.ndim > 1:
        a = a[:, 0]
    if sr != SR:
        raise ValueError(f"{path}: {sr} Hz, erwartet {SR}")
    return a


def detections(probs: list[float], cutoff: float, window: int) -> list[int]:
    hits, buf = [], []
    for i, p in enumerate(probs):
        buf.append(p)
        if len(buf) > window:
            buf.pop(0)
        if len(buf) == window and sum(buf) / window > cutoff:
            hits.append(i)
            buf.clear()
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", help="für sliding_window_size (Default: <model>.json oder 10)")
    ap.add_argument("--cutoffs", default="0.95,0.97,0.98,0.99")
    ap.add_argument("--pos", action="append", default=[])
    ap.add_argument("--neg", action="append", default=[], help="name=verzeichnis")
    ap.add_argument("--json", help="Ergebnis zusätzlich als JSON speichern")
    a = ap.parse_args()

    window = 10
    man = Path(a.manifest) if a.manifest else Path(a.model).with_name(Path(a.model).stem + "_manifest.json")
    if man.exists():
        window = json.loads(man.read_text()).get("micro", {}).get("sliding_window_size", window)
    cutoffs = [float(c) for c in a.cutoffs.split(",")]
    silence = np.zeros(SR, dtype=np.int16)
    result = {"model": a.model, "window": window, "recall": {}, "fa": {}}

    for pdir in a.pos:
        files = sorted(Path(pdir).glob("*.wav"))
        maxavg = []
        for f in files:
            probs = Model(a.model).predict_clip(np.concatenate([silence, load(f), silence]), step_ms=10)
            avgs = np.convolve(probs, np.ones(window) / window, mode="valid") if len(probs) >= window else [0]
            maxavg.append(float(np.max(avgs)))
        maxavg = np.array(maxavg)
        rec = {str(c): round(float(np.mean(maxavg > c)) * 100, 1) for c in cutoffs}
        result["recall"][pdir] = {"n": len(files), **rec}
        print(f"Erkennung {pdir} (n={len(files)}): " + "  ".join(f"@{c}: {rec[str(c)]}%" for c in cutoffs))

    for spec in a.neg:
        name, ndir = spec.split("=", 1)
        files = sorted(Path(ndir).glob("*.wav"))
        model = Model(a.model)
        probs, owner = [], []
        for f in files:
            p = model.predict_clip(np.concatenate([load(f), silence]), step_ms=10)
            probs += p
            owner += [f.name] * len(p)
        hours = len(probs) * 0.03 / 3600
        res = {"files": len(files), "hours": round(hours, 3)}
        for c in cutoffs:
            hits = detections(probs, c, window)
            res[str(c)] = {"fa": len(hits), "fa_per_h": round(len(hits) / hours, 2) if hours else None,
                           "files": sorted({owner[i] for i in hits})}
        result["fa"][name] = res
        print(f"Fehlalarme {name} ({len(files)} Dateien, {hours * 60:.1f} min): "
              + "  ".join(f"@{c}: {res[str(c)]['fa']}" for c in cutoffs))

    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
