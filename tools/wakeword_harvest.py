#!/usr/bin/env python3
"""Wake-Word-Harvest: bewertet jede Dobby-Satelliten-Auslösung automatisch.

Fragt alle POLL_S Sekunden `assist_pipeline/pipeline_debug/list` ab (HA hält nur
die letzten ~10 Läufe im Speicher, Warmups alle 30 Min eingeschlossen) und
schreibt für jeden neuen Satelliten-Lauf eine Zeile nach labels.jsonl:

  verdict = "real"       lokal erkannter Befehl oder LLM hat ein Tool aufgerufen
            "fp"         STT leer, oder LLM antwortet „Nichts erkannt.“ ohne Tool
            "unclear"    LLM antwortet ohne Aktion (Smalltalk, Frage, …)

Die WAV-Mitschnitte bleiben auf HA (/share/assist_debug/<device>/Dobby/<ns>/).
Abgeholt und zugeordnet werden sie erst vor einem Training vom Mac aus mit
tools/harvest_sync.py (Zuordnung über die Zeit, Ordnername ist monotonic_ns).

Aufruf:  HA_TOKEN=... HA_PIPELINE_ID=... HA_WS_URL=ws://<ha>:8123/api/websocket \
         python3 wakeword_harvest.py [--out /opt/mww-logger/harvest]
"""
import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import websockets

PIPELINE_ID = os.environ.get("HA_PIPELINE_ID", "")  # Assist-Pipeline-ID (z. B. Dobby)
POLL_S = 30
FP_PHRASES = ("nichts erkannt",)


def classify(events: list[dict]) -> dict | None:
    """Fasst einen Pipeline-Lauf zusammen; None für Läufe ohne Audio (Warmup/Text)."""
    by_type: dict[str, list[dict]] = {}
    for e in events:
        by_type.setdefault(e["type"], []).append(e)
    if "stt-start" not in by_type:
        return None

    start = by_type["run-start"][0]
    stt_text = ""
    if "stt-end" in by_type:
        stt_text = (by_type["stt-end"][0]["data"] or {}).get("stt_output", {}).get("text", "").strip()
    vad = {}
    for k in ("stt-vad-start", "stt-vad-end"):
        if k in by_type:
            vad[k] = (by_type[k][0]["data"] or {}).get("timestamp")

    speech, local, tool_calls = "", False, []
    if "intent-end" in by_type:
        d = by_type["intent-end"][0]["data"] or {}
        local = bool(d.get("processed_locally"))
        resp = (d.get("intent_output") or {}).get("response") or {}
        speech = ((resp.get("speech") or {}).get("plain") or {}).get("speech", "")
    for e in by_type.get("intent-progress", []):
        delta = (e.get("data") or {}).get("chat_log_delta") or {}
        for tc in delta.get("tool_calls") or []:
            tool_calls.append(tc.get("tool_name") or tc.get("name") or "?")

    errors = [(e.get("data") or {}).get("code") for e in by_type.get("error", [])]

    if local or tool_calls:
        verdict = "real"
    elif not stt_text.strip(" .…"):
        verdict = "fp"
    elif any(p in speech.lower() for p in FP_PHRASES):
        verdict = "fp"
    else:
        verdict = "unclear"

    return {
        "run_start": start["timestamp"],
        "stt_text": stt_text,
        "speech": speech,
        "processed_locally": local,
        "tool_calls": tool_calls,
        "vad_ms": vad,
        "errors": errors,
        "verdict": verdict,
    }


class Harvester:
    def __init__(self, url: str, token: str, out: Path):
        self.url, self.token, self.out = url, token, out
        self.out.mkdir(parents=True, exist_ok=True)
        self.labels = out / "labels.jsonl"
        self.seen_file = out / "seen_runs.txt"
        self.seen = set(self.seen_file.read_text().split()) if self.seen_file.exists() else set()
        self._id = 0

    async def call(self, ws, msg: dict) -> dict:
        self._id += 1
        msg["id"] = self._id
        await ws.send(json.dumps(msg))
        while True:
            r = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
            if r.get("id") == self._id:
                if not r.get("success", False):
                    raise RuntimeError(r)
                return r["result"]

    async def poll_once(self, ws) -> int:
        runs = (await self.call(ws, {"type": "assist_pipeline/pipeline_debug/list",
                                     "pipeline_id": PIPELINE_ID}))["pipeline_runs"]
        new = 0
        for r in runs:
            rid = r["pipeline_run_id"]
            if rid in self.seen:
                continue
            events = (await self.call(ws, {"type": "assist_pipeline/pipeline_debug/get",
                                           "pipeline_id": PIPELINE_ID,
                                           "pipeline_run_id": rid}))["events"]
            # Laufende Läufe noch nicht bewerten (kein run-end/error)
            if not any(e["type"] in ("run-end", "error") for e in events):
                continue
            row = classify(events)
            if row is not None:
                row["run_id"] = rid
                row["harvested_at"] = datetime.now(timezone.utc).isoformat()
                with self.labels.open("a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"{row['run_start']} {row['verdict']:8s} | {row['stt_text']!r} -> {row['speech']!r}", flush=True)
                new += 1
            self.seen.add(rid)
            with self.seen_file.open("a") as f:
                f.write(rid + "\n")
        return new

    async def run(self):
        while True:
            try:
                async with websockets.connect(self.url, max_size=32 * 1024 * 1024) as ws:
                    assert json.loads(await ws.recv())["type"] == "auth_required"
                    await ws.send(json.dumps({"type": "auth", "access_token": self.token}))
                    if json.loads(await ws.recv()).get("type") != "auth_ok":
                        print("Auth fehlgeschlagen", file=sys.stderr, flush=True)
                        await asyncio.sleep(300)
                        continue
                    print("verbunden", flush=True)
                    while True:
                        await self.poll_once(ws)
                        await asyncio.sleep(POLL_S)
            except Exception as exc:  # Netz/HA-Neustart: neu verbinden
                print(f"Verbindung verloren: {exc!r}", file=sys.stderr, flush=True)
                await asyncio.sleep(15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("HA_WS_URL"))
    ap.add_argument("--out", default="/opt/mww-logger/harvest")
    ap.add_argument("--once", action="store_true", help="einmal abfragen und beenden (Test)")
    a = ap.parse_args()
    token = os.environ.get("HA_TOKEN") or sys.exit("HA_TOKEN fehlt")
    if not a.url or not PIPELINE_ID:
        sys.exit("HA_WS_URL (oder --url) und HA_PIPELINE_ID müssen gesetzt sein")
    h = Harvester(a.url, token, Path(a.out))
    if a.once:
        async def once():
            async with websockets.connect(a.url, max_size=32 * 1024 * 1024) as ws:
                await ws.recv()
                await ws.send(json.dumps({"type": "auth", "access_token": token}))
                await ws.recv()
                print(await h.poll_once(ws), "neue Läufe")
        asyncio.run(once())
    else:
        asyncio.run(h.run())


if __name__ == "__main__":
    main()
