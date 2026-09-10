#!/usr/bin/env python3
"""
Wake Word Trainer for Home Assistant — macOS ARM (Apple Silicon / MPS)

Uses macOS built-in TTS voices + ffmpeg for sample generation.
No Linux dependencies, no Docker, no API keys needed.

Usage:
    python train.py "Hey Dobbi"
    python train.py "Hey Jarvis" --samples 500 --steps 5000
    python train.py "Hey Dobbi"  --full        # production quality
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import torch
import yaml
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

_IS_LINUX = sys.platform == "linux"

HF_FEATURES_URL = (
    "https://huggingface.co/datasets/davidscripka/openwakeword_features"
    "/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
)
HF_VALIDATION_URL = (
    "https://huggingface.co/datasets/davidscripka/openwakeword_features"
    "/resolve/main/validation_set_features.npy"
)

# English voices built into macOS — varied accents & tones
MACOS_VOICES = [
    "Samantha",       # US female (clear, neutral)
    "Alex",           # US male (default)
    "Fred",           # US male (old)
    "Albert",         # US male (nasal)
    "Daniel",         # GB male
    "Karen",          # AU female
    "Moira",          # IE female
    "Tessa",          # ZA female
    "Rishi",          # IN male
    "Aman",           # IN male
    "Eddy (Englisch (USA))",
    "Flo (Englisch (USA))",
    "Reed (Englisch (USA))",
    "Rocko (Englisch (USA))",
    "Sandy (Englisch (USA))",
    "Shelley (Englisch (USA))",
    "Grandma (Englisch (USA))",
    "Grandpa (Englisch (USA))",
    "Eddy (Englisch (UK))",
    "Reed (Englisch (UK))",
    "Rocko (Englisch (UK))",
    "Sandy (Englisch (UK))",
    "Shelley (Englisch (UK))",
]

# Speech rates (words per minute)
RATES = [140, 160, 180, 200, 220, 240]

# Pitch shifts in semitones (applied by ffmpeg after TTS)
PITCH_SHIFTS = [-2, -1, 0, 1, 2]

# Piper TTS speed variation (atempo filter multipliers)
RATE_FACTORS = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1]

# Piper voice models downloaded from rhasspy/piper-voices on HuggingFace
PIPER_VOICES_DIR = BASE_DIR / "piper-voices"
PIPER_VOICE_MODELS = [
    ("en_US-lessac-medium",      "en/en_US/lessac/medium"),
    ("en_US-ryan-medium",        "en/en_US/ryan/medium"),
    ("en_US-amy-medium",         "en/en_US/amy/medium"),
    ("en_US-joe-medium",         "en/en_US/joe/medium"),
    ("en_GB-jenny_dioco-medium", "en/en_GB/jenny_dioco/medium"),
    ("en_GB-alan-medium",        "en/en_GB/alan/medium"),
]

# German Piper voice models for German wake words (Linux/Docker)
PIPER_GERMAN_VOICE_MODELS = [
    ("de_DE-thorsten-medium",           "de/de_DE/thorsten/medium"),
    ("de_DE-thorsten_emotional-medium", "de/de_DE/thorsten_emotional/medium"),
    ("de_DE-kerstin-low",               "de/de_DE/kerstin/low"),
    ("de_DE-ramona-low",                "de/de_DE/ramona/low"),
    ("de_DE-karlsson-low",              "de/de_DE/karlsson/low"),
]

# German macOS TTS voices (de_DE) — used when training German wake words
MACOS_GERMAN_VOICES = [
    "Anna",
    "Eddy (Deutsch (Deutschland))",
    "Flo (Deutsch (Deutschland))",
    "Grandma (Deutsch (Deutschland))",
    "Grandpa (Deutsch (Deutschland))",
    "Reed (Deutsch (Deutschland))",
    "Rocko (Deutsch (Deutschland))",
    "Sandy (Deutsch (Deutschland))",
    "Shelley (Deutsch (Deutschland))",
]

# Phonetically similar phrases used as hard negatives during microWakeWord training.
# These prevent false positives from words that sound like the wake word.
MWW_CONFUSABLE_PHRASES = [
    # Closest rhymes to "Hey Dobbi" (differ by ≤1 phoneme — highest priority)
    "Hey Bobby", "Hey Bobbi", "Hey Tobi", "Hey Toby", "Hey Robbi",
    "Hey Robby", "Hey Hobby", "Hey Dobby", "Hey Poppi", "Hey Domi",
    "Hey Otti", "Hey Molli", "Hey Koby", "Hey Tommy", "Hey Bibi", "Hey Robin",
    # Name alone — forces "Hey" to be part of the required pattern
    "Dobbi", "Hey du",
    # Other smart home assistants heard in German households
    "Hey Siri", "Okay Google", "Alexa", "Hey Google", "Computer",
    # Short German interjections / filler openings
    "Hej", "Hallo", "Hör mal", "He du", "Hey Leute",
    "Ey", "Hey Mann", "Hey Mama", "Hey Papa",
]

# Everyday German sentences used as TTS adversarial negatives. Generated with the
# SAME voices as the positives, so the model cannot associate voice identity or
# TTS artifacts with the wake word — it must key on the phrase itself.
# (TTS adversarial samples beat other negative-mining methods: arXiv:2201.00167)
MWW_ADVERSARIAL_SENTENCES = [
    "Kannst du mir mal kurz helfen",
    "Was gibt es heute zu essen",
    "Ich gehe kurz einkaufen",
    "Wo ist die Fernbedienung schon wieder",
    "Komm mal bitte her",
    "Das war wirklich lecker",
    "Hast du die Tür abgeschlossen",
    "Morgen müssen wir früh aufstehen",
    "Der Hund muss noch mal raus",
    "Was läuft denn heute im Fernsehen",
    "Ich habe heute total viel zu tun",
    "Kommst du gleich mal in die Küche",
    "Das Wetter soll morgen besser werden",
    "Habt ihr schon Hausaufgaben gemacht",
    "Wir müssen noch den Müll rausbringen",
    "Gibst du mir mal das Salz rüber",
    "Ich bin gleich wieder da",
    "Mach doch bitte die Musik leiser",
    "Wann kommt Papa heute nach Hause",
    "Die Spülmaschine ist fertig",
    # Confusable words embedded in natural sentences (harder than the word alone)
    "Tobi kommt heute später nach Hause",
    "Wir fahren am Samstag zum Obi",
    "Das neue Hobby macht richtig Spaß",
    "Bobby hat schon wieder gebellt",
    "Die Lobby war gestern total voll",
    "Gabi hat vorhin angerufen",
    "Ich habe das Abi damals knapp bestanden",
    "Robbie spielt draußen im Garten",
    "Ey Mann das gibt es doch nicht",
    "Hey hast du das gesehen",
    "Hallo ist da jemand",
    "Okay das machen wir so",
    "Der Zombie-Film war richtig gruselig",
    "Toby und Domi kommen zum Essen",
    "Das Baby schläft endlich",
    "Habibi komm mal her",
    "Die Kollegen von Tommy waren dabei",
    "Deine Bobbahn steht noch im Keller",
]


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── MPS patch ─────────────────────────────────────────────────────────────────

def patch_openwakeword_mps():
    try:
        import openwakeword
        train_file = Path(openwakeword.__file__).parent / "train.py"
        content = train_file.read_text()

        old = "torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')"
        new = (
            "torch.device(\n"
            "            'cuda:0' if torch.cuda.is_available()\n"
            "            else 'mps' if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())\n"
            "            else 'cpu'\n"
            "        )"
        )
        if old in content:
            content = content.replace(old, new)
            train_file.write_text(content)
            print("  ✓ Patched openWakeWord for Apple MPS")
        else:
            print("  ✓ openWakeWord MPS patch: already applied or different version")
    except Exception as e:
        print(f"  ⚠ MPS patch skipped: {e}")


# ── Piper TTS (Linux / Docker) ────────────────────────────────────────────────

def find_piper() -> str:
    for candidate in ["/usr/local/piper/piper", str(BASE_DIR / "piper" / "piper"), "piper"]:
        if Path(candidate).is_file() or shutil.which(candidate):
            return candidate
    sys.exit("ERROR: piper binary not found. See README for installation instructions.")


def download_piper_voices() -> list[str]:
    """Download Piper voice models from HuggingFace on first run."""
    from huggingface_hub import hf_hub_download
    PIPER_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    available = []
    for model_name, hf_subpath in PIPER_VOICE_MODELS:
        onnx_file = PIPER_VOICES_DIR / f"{model_name}.onnx"
        json_file = PIPER_VOICES_DIR / f"{model_name}.onnx.json"
        try:
            if not onnx_file.exists():
                print(f"    Downloading voice: {model_name} (~60 MB)...")
                src = hf_hub_download("rhasspy/piper-voices",
                                      f"{hf_subpath}/{model_name}.onnx")
                shutil.copy(src, onnx_file)
                src_cfg = hf_hub_download("rhasspy/piper-voices",
                                          f"{hf_subpath}/{model_name}.onnx.json")
                shutil.copy(src_cfg, json_file)
            available.append(str(onnx_file))
        except Exception as e:
            tqdm.write(f"    ⚠ Skipped {model_name}: {e}")
    if not available:
        sys.exit("ERROR: No piper voice models could be downloaded. Check network access.")
    return available


def piper_to_wav(text: str, model_path: str, rate_factor: float, pitch_shift: int,
                 output_wav: Path, piper: str, ffmpeg: str):
    """Piper TTS → 16kHz mono WAV with speed and pitch variation."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        raw_wav = tmp.name
    try:
        subprocess.run(
            f"echo {shlex.quote(text)} | {shlex.quote(piper)} "
            f"--model {shlex.quote(model_path)} --output_file {shlex.quote(raw_wav)}",
            shell=True, check=True, capture_output=True,
        )
        _trim = "silenceremove=start_periods=1:start_threshold=-40dB,areverse,silenceremove=start_periods=1:start_threshold=-40dB,areverse"
        if rate_factor == 1.0 and pitch_shift == 0:
            af = f"{_trim},aresample=16000"
        elif pitch_shift == 0:
            af = f"{_trim},atempo={rate_factor:.3f},aresample=16000"
        else:
            # Normalize to 16k first — Piper voices output 22.05k, so the
            # asetrate trick must not assume the input rate
            factor = 2 ** (pitch_shift / 12) * rate_factor
            src_rate = int(16000 * factor)
            af = f"{_trim},aresample=16000,asetrate={src_rate},aresample=16000"
        subprocess.run(
            [ffmpeg, "-y", "-i", raw_wav,
             "-af", af, "-ar", "16000", "-ac", "1",
             "-acodec", "pcm_s16le", str(output_wav)],
            check=True, capture_output=True,
        )
    finally:
        try:
            os.unlink(raw_wav)
        except OSError:
            pass


# ── TTS sample generation ─────────────────────────────────────────────────────

def find_ffmpeg() -> str:
    for candidate in ["ffmpeg", "/opt/homebrew/bin/ffmpeg",
                      "/opt/homebrew/Caskroom/miniconda/base/bin/ffmpeg",
                      "/usr/local/bin/ffmpeg"]:
        if shutil.which(candidate):
            return candidate
    sys.exit("ERROR: ffmpeg not found. Install with: brew install ffmpeg")


def say_to_wav(text: str, voice: str, rate: int, pitch_shift: int,
               output_wav: Path, ffmpeg: str):
    """macOS say → AIFF → ffmpeg → 16kHz mono WAV with optional pitch shift."""
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
        aiff_path = tmp.name

    try:
        # Step 1: TTS → AIFF
        subprocess.run(
            ["say", "-v", voice, "-r", str(rate), "-o", aiff_path, text],
            check=True, capture_output=True,
        )

        # Step 2: AIFF → 16kHz mono WAV (+ optional pitch shift)
        # Trim leading/trailing silence first — training windows are END-aligned
        # (jitter 0.2s + truncate_start keeps the last 1.5s), so trailing TTS
        # silence would push the wake word out of the training window.
        _trim = "silenceremove=start_periods=1:start_threshold=-40dB,areverse,silenceremove=start_periods=1:start_threshold=-40dB,areverse"
        if pitch_shift == 0:
            audio_filter = f"{_trim},aresample=16000"
        else:
            # pitch shift in semitones: multiply sample rate
            factor = 2 ** (pitch_shift / 12)
            src_rate = int(16000 * factor)
            audio_filter = f"{_trim},aresample=16000,asetrate={src_rate},aresample=16000"

        subprocess.run(
            [ffmpeg, "-y", "-i", aiff_path,
             "-af", audio_filter,
             "-ar", "16000", "-ac", "1",
             "-acodec", "pcm_s16le", str(output_wav)],
            check=True, capture_output=True,
        )
    finally:
        os.unlink(aiff_path)


# ── Piper Python backend (neural TTS, works on macOS too) ────────────────────
# The bundled piper binary only runs on Linux/Docker; on macOS the fallback was
# the robotic `say` voices. The piper-tts Python package runs everywhere and
# unlocks the de_DE-mls model with 236 real German speakers.

PIPER_PY_DE_VOICES = [
    ("de_DE-thorsten-high",             "de/de_DE/thorsten/high"),
    ("de_DE-thorsten_emotional-medium", "de/de_DE/thorsten_emotional/medium"),
    ("de_DE-eva_k-x_low",               "de/de_DE/eva_k/x_low"),
    ("de_DE-karlsson-low",              "de/de_DE/karlsson/low"),
    ("de_DE-kerstin-low",               "de/de_DE/kerstin/low"),
    ("de_DE-pavoque-low",               "de/de_DE/pavoque/low"),
    ("de_DE-ramona-low",                "de/de_DE/ramona/low"),
    ("de_DE-mls-medium",                "de/de_DE/mls/medium"),   # multi-speaker: 236 voices
]

_PIPER_PY_VOICE_CACHE: dict = {}


def _piper_py_available() -> bool:
    try:
        import piper  # noqa: F401
        return True
    except ImportError:
        return False


def download_piper_py_de_pool() -> list[tuple[str, int | None]]:
    """Download German Piper voices and expand multi-speaker models into a
    (onnx_path, speaker_id) pool. Returns [] if piper-tts is not installed."""
    if not _piper_py_available():
        return []
    import json as _json
    from huggingface_hub import hf_hub_download
    PIPER_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    pool: list[tuple[str, int | None]] = []
    for model_name, hf_subpath in PIPER_PY_DE_VOICES:
        onnx_file = PIPER_VOICES_DIR / f"{model_name}.onnx"
        json_file = PIPER_VOICES_DIR / f"{model_name}.onnx.json"
        try:
            if not onnx_file.exists() or not json_file.exists():
                print(f"    Downloading German voice: {model_name} …")
                shutil.copy(hf_hub_download("rhasspy/piper-voices",
                                            f"{hf_subpath}/{model_name}.onnx"), onnx_file)
                shutil.copy(hf_hub_download("rhasspy/piper-voices",
                                            f"{hf_subpath}/{model_name}.onnx.json"), json_file)
            n_spk = _json.load(open(json_file)).get("num_speakers", 1)
            if n_spk and n_spk > 1:
                pool.extend((str(onnx_file), sid) for sid in range(n_spk))
            else:
                pool.append((str(onnx_file), None))
        except Exception as e:
            tqdm.write(f"    ⚠ Skipped voice {model_name}: {e}")
    return pool


def piper_py_to_wav(text: str, onnx_path: str, speaker_id: int | None,
                    rate_factor: float, pitch_shift: int,
                    output_wav: Path, ffmpeg: str):
    """piper-tts (Python) → 16kHz mono WAV with speed and pitch variation."""
    import wave as _wave
    from piper import PiperVoice, SynthesisConfig

    voice = _PIPER_PY_VOICE_CACHE.get(onnx_path)
    if voice is None:
        voice = PiperVoice.load(onnx_path, onnx_path + ".json"
                                if Path(onnx_path + ".json").exists()
                                else str(Path(onnx_path).with_suffix(".onnx.json")))
        _PIPER_PY_VOICE_CACHE[onnx_path] = voice

    syn = SynthesisConfig(speaker_id=speaker_id,
                          length_scale=round(1.0 / rate_factor, 3))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        raw_wav = tmp.name
    try:
        with _wave.open(raw_wav, "wb") as w:
            voice.synthesize_wav(text, w, syn_config=syn)
        _trim = "silenceremove=start_periods=1:start_threshold=-40dB,areverse,silenceremove=start_periods=1:start_threshold=-40dB,areverse"
        if pitch_shift == 0:
            af = f"{_trim},aresample=16000"
        else:
            src_rate = int(16000 * 2 ** (pitch_shift / 12))
            af = f"{_trim},aresample=16000,asetrate={src_rate},aresample=16000"
        subprocess.run(
            [ffmpeg, "-y", "-i", raw_wav,
             "-af", af, "-ar", "16000", "-ac", "1",
             "-acodec", "pcm_s16le", str(output_wav)],
            check=True, capture_output=True,
        )
    finally:
        os.unlink(raw_wav)


def get_available_voices() -> list[str]:
    """Filter to only English voices actually installed on this Mac."""
    result = subprocess.run(["say", "-v", "?"], capture_output=True, text=True)
    installed = result.stdout + result.stderr
    available = []
    for v in MACOS_VOICES:
        search = v.split()[0]
        if search.lower() in installed.lower():
            available.append(v)
    return available if available else ["Samantha"]


def get_available_german_voices() -> list[str]:
    """Filter to only German (de_DE) voices actually installed on this Mac."""
    result = subprocess.run(["say", "-v", "?"], capture_output=True, text=True)
    installed = result.stdout + result.stderr
    available = []
    for v in MACOS_GERMAN_VOICES:
        if v.lower() in installed.lower():
            available.append(v)
    return available


def download_piper_german_voices() -> list[str]:
    """Download German Piper voice models from HuggingFace on first run."""
    from huggingface_hub import hf_hub_download
    PIPER_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    available = []
    for model_name, hf_subpath in PIPER_GERMAN_VOICE_MODELS:
        onnx_file = PIPER_VOICES_DIR / f"{model_name}.onnx"
        json_file = PIPER_VOICES_DIR / f"{model_name}.onnx.json"
        try:
            if not onnx_file.exists():
                print(f"    Downloading German voice: {model_name} (~60 MB)...")
                src = hf_hub_download("rhasspy/piper-voices",
                                      f"{hf_subpath}/{model_name}.onnx")
                shutil.copy(src, onnx_file)
                src_cfg = hf_hub_download("rhasspy/piper-voices",
                                          f"{hf_subpath}/{model_name}.onnx.json")
                shutil.copy(src_cfg, json_file)
            available.append(str(onnx_file))
        except Exception as e:
            tqdm.write(f"    ⚠ Skipped German voice {model_name}: {e}")
    return available


def generate_samples(
    wake_word: str,
    output_dir: Path,
    n_samples: int,
    val_split: float = 0.1,
    lang: str = "de",
) -> tuple[Path, Path]:
    """Generate TTS samples. Uses Piper TTS on Linux/Docker, macOS say on macOS.
    When lang='de', prefers German voices (2:1 ratio German:English)."""
    ffmpeg = find_ffmpeg()

    train_dir = output_dir / "positive_train"
    test_dir = output_dir / "positive_test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    existing_train = len(list(train_dir.glob("*.wav")))
    existing_test = len(list(test_dir.glob("*.wav")))
    if existing_train + existing_test >= n_samples:
        print(f"  ✓ {existing_train + existing_test} samples already exist, skipping")
        return train_dir, test_dir

    n_val = max(int(n_samples * val_split), min(20, n_samples // 5))

    # Prosody variants only — NO trailing words: training windows are
    # end-aligned, extra words after the wake word would shift it out
    text_variants = [
        wake_word,
        wake_word + ".",
        wake_word + "!",
        wake_word + "?",
        wake_word.lower(),
    ]

    failed = 0

    # Sanity bounds for a spoken wake phrase — TTS models (esp. multi-speaker
    # mls) can hallucinate long babble on very short prompts. Such clips would
    # teach "any speech = wake word" → false positives.
    n_words = max(1, len(wake_word.split()))
    # Cap at 1.75s so every positive fits the 2000ms training window minus jitter
    min_dur_s, max_dur_s = 0.35, min(1.2 + 0.6 * n_words, 1.75)

    def _dur_ok(p: Path) -> bool:
        try:
            dur = max(0, p.stat().st_size - 44) / 32000   # 16k mono s16le
        except OSError:
            return False
        return min_dur_s <= dur <= max_dur_s

    if _IS_LINUX:
        piper_bin = find_piper()
        if lang == "de":
            de_models = download_piper_german_voices()
            en_models = download_piper_voices()
            # 2:1 German:English ratio
            voice_models = (de_models * 2 + en_models) if de_models else en_models
        else:
            voice_models = download_piper_voices()
        print(f"  Using {len(voice_models)} Piper voices × {len(RATE_FACTORS)} speeds × {len(PITCH_SHIFTS)} pitches")
        combos = [
            (voice_models[i % len(voice_models)],
             RATE_FACTORS[i % len(RATE_FACTORS)],
             PITCH_SHIFTS[i % len(PITCH_SHIFTS)],
             text_variants[i % len(text_variants)])
            for i in range(n_samples)
        ]
        for idx, (model, rate_factor, pitch, text) in enumerate(tqdm(combos, desc="Generating TTS (Piper)")):
            dest_dir = test_dir if idx < n_val else train_dir
            out_wav = dest_dir / f"{uuid.uuid4().hex}.wav"
            try:
                piper_to_wav(text, model, rate_factor, pitch, out_wav, piper_bin, ffmpeg)
                if not _dur_ok(out_wav):
                    # Retry once with a different voice
                    alt = voice_models[(voice_models.index(model) + 1) % len(voice_models)]
                    piper_to_wav(text, alt, 1.0, 0, out_wav, piper_bin, ffmpeg)
                if not _dur_ok(out_wav):
                    out_wav.unlink(missing_ok=True)
                    failed += 1
            except Exception as e:
                failed += 1
                tqdm.write(f"  ⚠ Skipped sample {idx}: {e}")
    else:
        de_voices = get_available_german_voices() if lang == "de" else []
        en_voices = get_available_voices()
        if de_voices:
            # 2:1 German:English — German pronunciation is primary target
            voices = de_voices * 2 + en_voices
        else:
            voices = en_voices
        # Neural Piper voices (Python backend) — far better quality than `say`,
        # incl. the 236-speaker de_DE-mls model. 3 of 4 samples use Piper,
        # 1 of 4 keeps `say` for engine diversity.
        piper_pool = download_piper_py_de_pool() if lang == "de" else []
        if piper_pool:
            print(f"  Using {len(piper_pool)} Piper voices/speakers (3:1) + {len(voices)} macOS say voices")
        else:
            print(f"  Using {len(voices)} macOS voices ({len(de_voices)} de_DE + {len(en_voices)} en) × {len(RATES)} speeds × {len(PITCH_SHIFTS)} pitches")
        for idx in tqdm(range(n_samples), desc="Generating TTS"):
            dest_dir = test_dir if idx < n_val else train_dir
            out_wav = dest_dir / f"{uuid.uuid4().hex}.wav"
            pitch = PITCH_SHIFTS[idx % len(PITCH_SHIFTS)]
            text = text_variants[idx % len(text_variants)]
            try:
                if piper_pool and idx % 4 != 3:
                    onnx, spk = piper_pool[(idx * 7919) % len(piper_pool)]  # prime stride → spread speakers
                    rf = RATE_FACTORS[idx % len(RATE_FACTORS)]
                    piper_py_to_wav(text, onnx, spk, rf, pitch, out_wav, ffmpeg)
                else:
                    voice = voices[idx % len(voices)]
                    rate = RATES[idx % len(RATES)]
                    say_to_wav(text, voice, rate, pitch, out_wav, ffmpeg)
                if not _dur_ok(out_wav):
                    # Hallucinated/empty TTS — fall back to a reliable say voice
                    say_to_wav(text, voices[idx % len(voices)],
                               RATES[idx % len(RATES)], pitch, out_wav, ffmpeg)
                if not _dur_ok(out_wav):
                    out_wav.unlink(missing_ok=True)
                    failed += 1
            except Exception as e:
                failed += 1
                tqdm.write(f"  ⚠ Skipped sample {idx}: {e}")

    good = len(list(train_dir.glob("*.wav")))
    print(f"  ✓ {good} training + {len(list(test_dir.glob('*.wav')))} validation WAVs"
          + (f" ({failed} failed)" if failed else ""))
    return train_dir, test_dir


# ── Augmentation data ─────────────────────────────────────────────────────────

def _audio_to_16k_int16(audio_entry: dict) -> np.ndarray | None:
    """Convert a HuggingFace audio dict to 16kHz mono int16 numpy array.
    Works with datasets 2.x where entries have 'array' + 'sampling_rate' keys."""
    try:
        arr = np.array(audio_entry["array"], dtype=np.float32)
        sr = int(audio_entry.get("sampling_rate", 16000))

        if sr != 16000:
            import torchaudio
            waveform = torch.from_numpy(arr).unsqueeze(0)
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            arr = waveform.squeeze().numpy()

        if arr.ndim > 1:
            arr = arr.mean(axis=0)

        # Normalise to int16 range
        peak = np.abs(arr).max()
        if peak > 0:
            arr = arr / peak
        return (arr * 32767).astype(np.int16)
    except Exception:
        return None


def download_background_data(full_mode: bool):
    import datasets as hf_datasets

    rir_dir = DATA_DIR / "mit_rirs"
    if not rir_dir.exists():
        print("  Downloading MIT Room Impulse Responses (~50 MB)...")
        rir_dir.mkdir(parents=True)
        # Use trust_remote_code=False and no Audio cast to avoid torchcodec
        ds = hf_datasets.load_dataset(
            "davidscripka/MIT_environmental_impulse_responses",
            split="train", streaming=True,
        )
        ok = 0
        for row in tqdm(ds, desc="RIRs"):
            audio = row.get("audio", {})
            arr = _audio_to_16k_int16(audio)
            if arr is not None:
                name = (Path(audio.get("path", "")).name or f"{uuid.uuid4().hex}.wav")
                if not name.endswith(".wav"):
                    name = Path(name).stem + ".wav"
                scipy.io.wavfile.write(str(rir_dir / name), 16000, arr)
                ok += 1
        print(f"  ✓ RIRs: {ok} files")
    else:
        print(f"  ✓ RIRs present ({len(list(rir_dir.glob('*.wav')))} files)")

    audioset_dir = DATA_DIR / "audioset_16k"
    if not audioset_dir.exists():
        # Download ~500 clips from AudioSet balanced train via HuggingFace datasets
        n_clips = 2000 if full_mode else 500
        print(f"  Downloading AudioSet background noise ({n_clips} clips)...")
        audioset_dir.mkdir(parents=True)
        ds = hf_datasets.load_dataset(
            "agkphysics/AudioSet", "balanced",
            split="train", streaming=True, trust_remote_code=True,
        )
        ok = 0
        for i, row in enumerate(tqdm(ds, total=n_clips, desc="AudioSet")):
            if ok >= n_clips:
                break
            arr = _audio_to_16k_int16(row["audio"])
            if arr is not None:
                name = f"{row.get('video_id', uuid.uuid4().hex)}.wav"
                scipy.io.wavfile.write(str(audioset_dir / name), 16000, arr)
                ok += 1
        print(f"  ✓ AudioSet: {ok} files")
    else:
        print(f"  ✓ AudioSet present ({len(list(audioset_dir.glob('*.wav')))} files)")

    # Environmental sounds: ESC-50 (primary) + UrbanSound8k (fallback)
    # Replaces the defunct DynamicSuperb/BackgroundMusicAdding_MUSAN source.
    # Both datasets contain pure environmental/urban sounds without speech.
    env_dir = DATA_DIR / "musan"
    if not env_dir.exists() or not any(env_dir.rglob("*.wav")):
        n_clips = 200 if full_mode else 80
        print(f"  Downloading environmental sounds ({n_clips} clips)...")
        env_dir.mkdir(parents=True, exist_ok=True)
        ok = 0
        for dataset_id, split in [
            ("ashraq/esc50", "train"),
            ("danavery/urbansound8k", "train"),
        ]:
            if ok >= n_clips:
                break
            try:
                ds = hf_datasets.load_dataset(
                    dataset_id, split=split, streaming=True, trust_remote_code=True,
                )
                label = dataset_id.split("/")[1]
                for row in tqdm(ds, total=n_clips - ok, desc=label):
                    if ok >= n_clips:
                        break
                    audio = row.get("audio", {})
                    if not isinstance(audio, dict):
                        continue
                    arr = _audio_to_16k_int16(audio)
                    if arr is not None:
                        scipy.io.wavfile.write(str(env_dir / f"{uuid.uuid4().hex}.wav"), 16000, arr)
                        ok += 1
            except Exception as e:
                print(f"  ⚠ {dataset_id} failed ({e}), trying next source...")
        if ok > 0:
            print(f"  ✓ Environmental sounds: {ok} files")
        else:
            print(f"  ⚠ All environmental sound downloads failed, using AudioSet only")
    else:
        print(f"  ✓ Environmental sounds present ({len(list(env_dir.rglob('*.wav')))} files)")


def download_feature_data(full_mode: bool) -> str:
    import urllib.request

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    val_path = DATA_DIR / "validation_set_features.npy"
    if not val_path.exists():
        print("  Downloading validation features (~400 MB)...")
        urllib.request.urlretrieve(HF_VALIDATION_URL, val_path, _progress_hook("Val features"))
    else:
        print(f"  ✓ Validation features: {val_path.stat().st_size // 1024 // 1024} MB")

    if full_mode:
        acav_path = DATA_DIR / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
        if not acav_path.exists():
            print("  Downloading ACAV100M negative features (~11 GB, one-time)...")
            urllib.request.urlretrieve(HF_FEATURES_URL, acav_path, _progress_hook("ACAV100M"))
        else:
            print(f"  ✓ ACAV100M: {acav_path.stat().st_size // 1024 // 1024 // 1024} GB")
        return str(acav_path)
    else:
        print("  (Quick mode: validation set used as negatives — use --full for production quality)")
        return str(val_path)


def _progress_hook(label: str):
    last = [0]
    def hook(count, block_size, total_size):
        if total_size <= 0:
            return
        pct = min(count * block_size * 100 // total_size, 100)
        if pct - last[0] >= 10:
            print(f"    {label}: {pct}%", end="\r", flush=True)
            last[0] = pct
    return hook


# ── Training config ───────────────────────────────────────────────────────────

def make_config(wake_word: str, negative_features_path: str,
                full_mode: bool, steps: int, n_samples: int) -> tuple[Path, str]:
    model_name = (wake_word.lower()
                  .replace(" ", "_")
                  .replace(",", "")
                  .replace("!", "")
                  .replace(".", ""))

    config = {
        "model_name": model_name,
        "target_phrase": [wake_word],
        "custom_negative_phrases": [],
        "n_samples": n_samples,
        "n_samples_val": max(int(n_samples * 0.1), 50),
        "tts_batch_size": 50,
        "augmentation_batch_size": 16,
        "piper_sample_generator_path": str(BASE_DIR / "piper-sample-generator"),  # not used
        "output_dir": str(OUTPUT_DIR),
        "rir_paths": [str(DATA_DIR / "mit_rirs")],
        "background_paths": [
            p for p in [
                str(DATA_DIR / "audioset_16k"),
                str(DATA_DIR / "musan"),
                str(DATA_DIR / "custom_background"),
            ] if list(Path(p).rglob("*.wav")) or list(Path(p).rglob("*.flac"))
        ],
        "false_positive_validation_data_path": str(DATA_DIR / "validation_set_features.npy"),
        "augmentation_rounds": 2 if full_mode else 1,
        "feature_data_files": {
            "negative_features": negative_features_path,
        },
        "batch_n_per_class": {
            "negative_features": 512 if full_mode else 256,
            "adversarial_negative": 50,
            "positive": 50,
        },
        "model_type": "dnn",
        "layer_size": 32,
        "steps": steps,
        "max_negative_weight": 1000 if full_mode else 500,
        "target_false_positives_per_hour": 0.5,
        "target_accuracy": 0.5,
        "target_recall": 0.2,
    }

    config_path = BASE_DIR / f"{model_name}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path, model_name


# ── In-process augmentation + training ───────────────────────────────────────

def _apply_torchaudio_shims():
    """Patch torchaudio I/O for Python 3.13 / torchaudio 2.x (torchcodec not available)."""
    import soundfile as sf
    import torchaudio as ta

    def _load(path, *a, **kw):
        data, sr = sf.read(str(path), always_2d=True)
        return torch.from_numpy(data.T.copy()).float(), sr

    class _Meta:
        def __init__(self, p):
            i = sf.info(str(p))
            self.num_channels = i.channels
            self.sample_rate = i.samplerate
            self.num_frames = i.frames

    ta.load = _load
    ta.info = lambda p, *a, **kw: _Meta(p)
    if not hasattr(ta, "set_audio_backend"):
        ta.set_audio_backend = lambda *a, **kw: None


def _run_augment_and_train(config_path: Path, model_name: str, model_dir: Path,
                            steps: int, wake_word: str):
    """Run openWakeWord augmentation + training in-process with all compatibility patches."""
    import sys as _sys
    import os as _os

    # Add piper-sample-generator stub so openWakeWord's train.py import works
    stub_dir = str(BASE_DIR / "piper-sample-generator")
    if stub_dir not in _sys.path:
        _sys.path.insert(0, stub_dir)

    # Apply torchaudio shims BEFORE any openWakeWord imports
    _apply_torchaudio_shims()

    # Now import openWakeWord internals (patches are already in effect)
    import yaml as _yaml
    import numpy as np
    import scipy.io.wavfile
    from pathlib import Path as P

    sys_oww = str(BASE_DIR / "openWakeWord")
    if sys_oww not in _sys.path:
        _sys.path.insert(0, sys_oww)

    import openwakeword
    import openwakeword.utils
    from openwakeword.data import augment_clips
    from openwakeword.utils import compute_features_from_generator
    from openwakeword.train import Model

    config = _yaml.safe_load(open(config_path).read())

    # Resolve paths
    pos_train_dir = P(config["output_dir"]) / model_name / "positive_train"
    pos_test_dir  = P(config["output_dir"]) / model_name / "positive_test"
    neg_train_dir = P(config["output_dir"]) / model_name / "negative_train"
    neg_test_dir  = P(config["output_dir"]) / model_name / "negative_test"
    feature_dir   = P(config["output_dir"]) / model_name

    # Create negative dirs (empty — we have no custom negative phrases)
    neg_train_dir.mkdir(parents=True, exist_ok=True)
    neg_test_dir.mkdir(parents=True, exist_ok=True)

    rir_paths = [e.path for d in config["rir_paths"] for e in _os.scandir(d)]
    background_paths = []
    for bg_path in config["background_paths"]:
        try:
            background_paths.extend([e.path for e in _os.scandir(bg_path)])
        except FileNotFoundError:
            pass
    if not background_paths:
        sys.exit("ERROR: No background audio found. Run step 3 first.")

    # Determine clip duration from samples
    pos_clips_test = list(pos_test_dir.glob("*.wav"))
    durations = []
    for p in pos_clips_test[:50]:
        sr, d = scipy.io.wavfile.read(str(p))
        durations.append(len(d))
    total_length = max(32000, int(round(np.median(durations) / 1000) * 1000) + 12000)
    if abs(total_length - 32000) <= 4000:
        total_length = 32000

    batch_size = config.get("augmentation_batch_size", 16)
    rounds     = config.get("augmentation_rounds", 1)
    n_cpus     = max(1, (_os.cpu_count() or 2) // 2)

    def _feat_file(name):
        return str(feature_dir / name)

    # ── Step A: Augmentation → feature extraction ─────────────────────────────
    if not (feature_dir / "positive_features_train.npy").exists():
        print("  Step A: Augmenting clips + extracting features...")

        pos_train_clips = [str(p) for p in pos_train_dir.glob("*.wav")] * rounds
        pos_test_clips  = [str(p) for p in pos_test_dir.glob("*.wav")]  * rounds
        neg_train_clips = [str(p) for p in neg_train_dir.glob("*.wav")] * rounds
        neg_test_clips  = [str(p) for p in neg_test_dir.glob("*.wav")]  * rounds

        compute_features_from_generator(
            augment_clips(pos_train_clips, total_length=total_length, batch_size=batch_size,
                          background_clip_paths=background_paths, RIR_paths=rir_paths),
            n_total=len(pos_train_clips), clip_duration=total_length,
            output_file=_feat_file("positive_features_train.npy"),
            device="cpu", ncpu=n_cpus)

        compute_features_from_generator(
            augment_clips(pos_test_clips, total_length=total_length, batch_size=batch_size,
                          background_clip_paths=background_paths, RIR_paths=rir_paths),
            n_total=len(pos_test_clips), clip_duration=total_length,
            output_file=_feat_file("positive_features_test.npy"),
            device="cpu", ncpu=n_cpus)

        # Negative clips (empty if no custom_negative_phrases — fine, skip)
        if neg_train_clips:
            compute_features_from_generator(
                augment_clips(neg_train_clips, total_length=total_length, batch_size=batch_size,
                              background_clip_paths=background_paths, RIR_paths=rir_paths),
                n_total=len(neg_train_clips), clip_duration=total_length,
                output_file=_feat_file("negative_features_train.npy"),
                device="cpu", ncpu=n_cpus)
            compute_features_from_generator(
                augment_clips(neg_test_clips, total_length=total_length, batch_size=batch_size,
                              background_clip_paths=background_paths, RIR_paths=rir_paths),
                n_total=len(neg_test_clips), clip_duration=total_length,
                output_file=_feat_file("negative_features_test.npy"),
                device="cpu", ncpu=n_cpus)

        print("  ✓ Feature extraction done")
    else:
        print("  ✓ Features already present, skipping augmentation")

    # ── Step B: Train DNN ─────────────────────────────────────────────────────
    print(f"  Step B: Training DNN ({steps} steps)...")
    from openwakeword.data import mmap_batch_generator

    input_shape = np.load(_feat_file("positive_features_test.npy")).shape[1:]
    seconds_per_example = 1280 * input_shape[0] / 16000

    oww_model = Model(
        n_classes=1,
        input_shape=input_shape,
        model_type=config.get("model_type", "dnn"),
        layer_dim=config.get("layer_size", 32),
        seconds_per_example=seconds_per_example,
    )

    # Shape-reshape transform for background negative features
    def _reshape(x, n=input_shape[0]):
        if n != x.shape[1]:
            x = np.vstack(x)
            x = np.array([x[i:i+n, :] for i in range(0, x.shape[0]-n, n)])
        return x

    # Feature data files: main negatives from config + our positive clips
    feat_files = dict(config["feature_data_files"])
    feat_files["positive"] = _feat_file("positive_features_train.npy")

    data_transforms  = {k: _reshape for k in feat_files}
    label_transforms = {}
    for k in feat_files:
        label_transforms[k] = (lambda x: [1]*len(x)) if k == "positive" else (lambda x: [0]*len(x))

    # Adversarial negatives (only if we have any negative clips)
    has_adv_neg = (feature_dir / "negative_features_train.npy").exists()
    if has_adv_neg:
        feat_files["adversarial_negative"] = _feat_file("negative_features_train.npy")
        label_transforms["adversarial_negative"] = lambda x: [0]*len(x)

    batch_n = {k: v for k, v in config["batch_n_per_class"].items()
               if k != "adversarial_negative" or has_adv_neg}

    batch_gen = mmap_batch_generator(
        feat_files,
        n_per_class=batch_n,
        data_transform_funcs=data_transforms,
        label_transform_funcs=label_transforms,
    )

    class _IterDS(torch.utils.data.IterableDataset):
        def __iter__(self): return batch_gen

    X_train = torch.utils.data.DataLoader(
        _IterDS(), batch_size=None, num_workers=0
    )

    # False-positive validation data
    X_val_fp_arr = np.load(config["false_positive_validation_data_path"])
    X_val_fp_arr = np.array([X_val_fp_arr[i:i+input_shape[0]]
                              for i in range(0, X_val_fp_arr.shape[0]-input_shape[0], 1)])
    X_val_fp_lbl = np.zeros(X_val_fp_arr.shape[0], dtype=np.float32)
    X_val_fp = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_val_fp_arr), torch.from_numpy(X_val_fp_lbl)),
        batch_size=len(X_val_fp_lbl),
    )

    # Combined val set: pos + neg (use empty array if no adversarial negatives)
    val_pos = np.load(_feat_file("positive_features_test.npy"))
    if has_adv_neg:
        val_neg = np.load(_feat_file("negative_features_test.npy"))
    else:
        val_neg = np.zeros((0,) + val_pos.shape[1:], dtype=val_pos.dtype)

    val_labels = np.hstack((
        np.ones(val_pos.shape[0]), np.zeros(val_neg.shape[0])
    )).astype(np.float32)
    val_data = np.vstack((val_pos, val_neg)) if val_neg.shape[0] > 0 else val_pos
    X_val = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(val_data), torch.from_numpy(val_labels)),
        batch_size=len(val_labels),
    )

    best_model = oww_model.auto_train(
        X_train=X_train,
        X_val=X_val,
        false_positive_val_data=X_val_fp,
        steps=steps,
        max_negative_weight=config.get("max_negative_weight", 500),
        target_fp_per_hour=config.get("target_false_positives_per_hour", 0.5),
    )

    oww_model.export_model(
        model=best_model,
        model_name=model_name,
        output_dir=str(OUTPUT_DIR),
    )

    # Merge external data into a single self-contained ONNX file (required for HA openWakeWord add-on)
    _onnx_path = OUTPUT_DIR / f"{model_name}.onnx"
    if _onnx_path.exists():
        try:
            import onnx as _onnx
            import os as _os
            _prev_dir = _os.getcwd()
            _os.chdir(OUTPUT_DIR)
            _model = _onnx.load(f"{model_name}.onnx")
            # Rename input to x.1 to match official openWakeWord model format
            for _node in _model.graph.node:
                for _j, _inp in enumerate(_node.input):
                    if _inp == "x":
                        _node.input[_j] = "x.1"
            if _model.graph.input and _model.graph.input[0].name == "x":
                _model.graph.input[0].name = "x.1"
            _onnx.save_model(_model, f"{model_name}.onnx", save_as_external_data=False)
            _os.chdir(_prev_dir)
            # Remove leftover .onnx.data file if present
            _data_file = OUTPUT_DIR / f"{model_name}.onnx.data"
            if _data_file.exists():
                _data_file.unlink()
            print("  ✓ Merged ONNX external data + fixed input name (x→x.1)")
        except Exception as _e:
            print(f"  ⚠ Could not merge ONNX external data: {_e}")

    onnx_files = list(OUTPUT_DIR.glob(f"{model_name}*.onnx"))
    mf = onnx_files[0] if onnx_files else feature_dir / f"{model_name}.onnx"
    print(f"""
╔══════════════════════════════════════════════════╗
║                 Training Complete!               ║
╚══════════════════════════════════════════════════╝
  Model : {mf}
  Size  : {mf.stat().st_size // 1024 if mf.exists() else '?'} KB

  To install in Home Assistant:
  → Copy .onnx to /share/openwakeword/ on your HA host
  → Restart Wyoming openWakeWord add-on
  → Settings → Voice Assistants → Wake word → "{wake_word}"
""")


# ── microWakeWord helpers ──────────────────────────────────────────────────────

MWW_REPO_DIR = BASE_DIR / "microWakeWord"
MWW_NEG_DIR  = DATA_DIR / "mww_negative"

MWW_NEG_DATASETS = [
    ("speech",          "speech.zip"),
    ("no_speech",       "no_speech.zip"),
    ("dinner_party",    "dinner_party.zip"),
    ("dinner_party_eval", "dinner_party_eval.zip"),
]
MWW_HF_ROOT = "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/"


def _mww_repo_ok() -> bool:
    return (MWW_REPO_DIR / "microwakeword" / "audio").exists()


def _mww_sys_path():
    p = str(MWW_REPO_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def _download_mww_negatives():
    import urllib.request, zipfile
    MWW_NEG_DIR.mkdir(parents=True, exist_ok=True)
    for folder, fname in MWW_NEG_DATASETS:
        if (MWW_NEG_DIR / folder).exists():
            print(f"  ✓ {folder}")
            continue
        print(f"  Downloading {fname} …")
        zip_path = MWW_NEG_DIR / fname
        def _hook(count, bs, total):
            if total > 0:
                pct = min(100, count * bs * 100 // total)
                print(f"    {pct}%", end="\r", flush=True)
        urllib.request.urlretrieve(MWW_HF_ROOT + fname, str(zip_path), _hook)
        print()
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(MWW_NEG_DIR))
        zip_path.unlink()
        print(f"  ✓ {folder}")


# ── German speech negatives (Mozilla Common Voice, CC-0) ─────────────────────
# Closes the biggest FP gap: the kahrendt HF negatives are English-only, but
# household false triggers come from *German* everyday speech. We stream real
# German clips (thousands of speakers) from an ungated Common Voice mirror —
# no local recording needed.

GERMAN_CV_REPO           = "fsicoli/common_voice_22_0"                # ungated CV mirror, CC-0
GERMAN_SPEECH_DIR        = DATA_DIR / "german_speech"                 # training negatives (wav)
GERMAN_SPEECH_EVAL_DIR   = DATA_DIR / "german_speech_eval"            # held-out clips for ambient eval
GERMAN_FEATURES_DIR      = DATA_DIR / "german_speech_features"        # cached spectrograms (train)
GERMAN_EVAL_FEATURES_DIR = DATA_DIR / "german_speech_eval_features"   # cached ambient eval spectrograms


def _download_german_speech(n_train: int = 6000, n_eval: int = 400) -> bool:
    """Stream German Common Voice clips as wake-word negatives.

    Streams tar shards over HTTP and stops as soon as enough clips are
    converted — the full shard (several GB) is never stored on disk.
    Every 16th clip is held out for the German ambient eval set.
    Returns True if enough clips are available locally."""
    import hashlib, tarfile, tempfile, urllib.request

    ffmpeg = find_ffmpeg()
    GERMAN_SPEECH_DIR.mkdir(parents=True, exist_ok=True)
    GERMAN_SPEECH_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    def _counts():
        return (len(list(GERMAN_SPEECH_DIR.glob("*.wav"))),
                len(list(GERMAN_SPEECH_EVAL_DIR.glob("*.wav"))))

    have_train, have_eval = _counts()
    if have_train >= n_train and have_eval >= n_eval:
        print(f"  ✓ German speech negatives cached ({have_train} train / {have_eval} eval)")
        return True

    print(f"  Need {n_train} train + {n_eval} eval clips (have {have_train}/{have_eval})")
    shard = 0
    while have_train < n_train or have_eval < n_eval:
        url = (f"https://huggingface.co/datasets/{GERMAN_CV_REPO}"
               f"/resolve/main/audio/de/train/de_train_{shard}.tar")
        print(f"  Streaming Common Voice DE shard {shard} …")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "speaky-trainer"})
            resp = urllib.request.urlopen(req, timeout=120)
        except Exception as e:
            print(f"  ⚠ Could not open shard {shard}: {e}")
            break
        try:
            # mode="r|" reads the tar as a forward-only stream (no seeking, no full download)
            with tarfile.open(fileobj=resp, mode="r|*") as tar:
                for member in tar:
                    if not member.isfile() or not member.name.endswith(".mp3"):
                        continue
                    # Deterministic split per file name — stable across resumed runs
                    stem = Path(member.name).stem
                    is_eval = (int(hashlib.md5(stem.encode()).hexdigest(), 16) % 16 == 0)
                    dst_dir = GERMAN_SPEECH_EVAL_DIR if is_eval else GERMAN_SPEECH_DIR
                    dst = dst_dir / f"cv_de_{stem}.wav"
                    if dst.exists():
                        continue
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                        tmp.write(f.read())
                        tmp_path = tmp.name
                    try:
                        subprocess.run(
                            [ffmpeg, "-y", "-i", tmp_path, "-t", "9",
                             "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", str(dst)],
                            check=True, capture_output=True)
                        # Discard clips shorter than ~1.5 s (44-byte header + 32 kB/s)
                        if dst.stat().st_size < 48000:
                            dst.unlink()
                        elif is_eval:
                            have_eval += 1
                        else:
                            have_train += 1
                    except subprocess.CalledProcessError:
                        dst.unlink(missing_ok=True)
                    finally:
                        os.unlink(tmp_path)
                    if (have_train + have_eval) % 250 == 0:
                        print(f"    {have_train} train / {have_eval} eval …", flush=True)
                    if have_train >= n_train and have_eval >= n_eval:
                        break
        except Exception as e:
            print(f"  ⚠ Shard {shard} stream aborted: {e}")
        finally:
            resp.close()
        if have_train >= n_train and have_eval >= n_eval:
            break
        shard += 1
        if shard > 40:   # safety stop
            break

    have_train, have_eval = _counts()
    print(f"  ✓ German speech negatives: {have_train} train / {have_eval} eval clips")
    return have_train >= 500   # enough to be useful even if the target was missed


def _generate_german_eval_features() -> bool:
    """Build a German ambient eval set: concatenate held-out Common Voice clips
    into two long streams and generate un-augmented spectrograms in the same
    validation_ambient/testing_ambient layout as dinner_party_eval. This makes
    the reported false-accepts-per-hour metric include German speech."""
    _mww_sys_path()
    from microwakeword.audio.clips import Clips
    from microwakeword.audio.spectrograms import SpectrogramGeneration
    from mmap_ninja.ragged import RaggedMmap

    wavs = sorted(GERMAN_SPEECH_EVAL_DIR.glob("*.wav"))
    if len(wavs) < 20:
        return False
    ffmpeg = find_ffmpeg()
    # Rebuild everything if the held-out clip set grew since the cache was built
    count_file = GERMAN_EVAL_FEATURES_DIR / "_clip_count.txt"
    cached_count = int(count_file.read_text()) if count_file.exists() else -1
    if cached_count != len(wavs) and GERMAN_EVAL_FEATURES_DIR.exists():
        shutil.rmtree(str(GERMAN_EVAL_FEATURES_DIR))
    GERMAN_EVAL_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    count_file.write_text(str(len(wavs)))

    half = len(wavs) // 2
    for split_name, files in [("validation_ambient", wavs[:half]),
                              ("testing_ambient",    wavs[half:])]:
        mmap_dir = GERMAN_EVAL_FEATURES_DIR / split_name / "german_cv_mmap"
        if mmap_dir.exists() and (mmap_dir / "dtype.ninja").exists():
            print(f"  ✓ German eval {split_name} cached")
            continue
        if mmap_dir.exists():
            shutil.rmtree(str(mmap_dir))
        stream_dir = GERMAN_EVAL_FEATURES_DIR / f"_stream_{split_name}"
        stream_dir.mkdir(parents=True, exist_ok=True)
        stream_wav = stream_dir / "stream.wav"
        if not stream_wav.exists():
            concat_list = stream_dir / "concat.txt"
            concat_list.write_text("".join(f"file '{w.resolve()}'\n" for w in files))
            subprocess.run(
                [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
                 "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", str(stream_wav)],
                check=True, capture_output=True)
        print(f"  Generating German eval spectrograms ({split_name}) …")
        mmap_dir.parent.mkdir(parents=True, exist_ok=True)
        clips = Clips(input_directory=str(stream_dir), file_pattern="*.wav",
                      max_clip_duration_s=None, remove_silence=False,
                      random_split_seed=None)
        sg = SpectrogramGeneration(clips=clips, augmenter=None, step_ms=10)
        RaggedMmap.from_generator(
            out_dir=str(mmap_dir),
            sample_generator=sg.spectrogram_generator(),
            batch_size=10, verbose=False)
        print(f"  ✓ {split_name} done")
    return True


def _lufs_normalize_dir(src_dir: Path, dst_dir: Path, ffmpeg: str,
                         pattern: str = "*.wav", target_lufs: float = -23.0) -> int:
    """LUFS-normalize (EBU R128) WAVs from src_dir matching pattern into dst_dir.
    Returns number of files processed."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    wavs = list(src_dir.glob(pattern))
    count = 0
    for wav in wavs:
        dst = dst_dir / wav.name
        if dst.exists():
            continue
        try:
            # Normalize FIRST, then trim edge silence — recordings can be very
            # quiet, so an absolute trim threshold only works after loudnorm.
            # Trailing silence would otherwise push the word out of the
            # end-aligned training window.
            subprocess.run(
                [ffmpeg, "-y", "-i", str(wav),
                 "-af", (f"loudnorm=I={target_lufs}:TP=-2:LRA=7,aresample=16000,"
                         "silenceremove=start_periods=1:start_threshold=-40dB,"
                         "areverse,silenceremove=start_periods=1:start_threshold=-40dB,areverse"),
                 "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", str(dst)],
                check=True, capture_output=True,
            )
            # Discard files that ended up empty/too short after trimming
            if dst.exists() and dst.stat().st_size < 44 + int(0.3 * 32000):
                dst.unlink()
                continue
            count += 1
        except subprocess.CalledProcessError:
            pass
    return count


def _generate_confusable_tts(phrases: list[str], out_dir: Path, n_per_phrase: int = 6) -> None:
    """Generate TTS WAVs for confusable/similar-sounding phrases (hard negatives).
    Uses German voices (primary) + English voices on macOS; English Piper on Linux."""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("*.wav")))
    expected = len(phrases) * n_per_phrase
    if existing >= expected:
        print(f"  ✓ {existing} confusable WAVs cached")
        return
    ffmpeg = find_ffmpeg()
    pitch_shifts = [-1, 0, 1, -2, 0, 2]
    idx = existing
    if _IS_LINUX:
        piper = find_piper()
        # German voices first — these phrases are German
        voices = (download_piper_german_voices() or []) * 2 + download_piper_voices()
        rate_factors = [0.9, 1.0, 1.1, 0.95, 1.05, 0.85]
        for phrase in phrases:
            for k in range(n_per_phrase):
                out_wav = out_dir / f"conf_{idx:04d}.wav"
                if not out_wav.exists():
                    model = voices[k % len(voices)]
                    rate = rate_factors[k % len(rate_factors)]
                    pitch = pitch_shifts[k % len(pitch_shifts)]
                    try:
                        piper_to_wav(phrase, model, rate, pitch, out_wav, piper, ffmpeg)
                    except Exception:
                        pass
                idx += 1
    else:
        de_voices = get_available_german_voices()
        en_voices = get_available_voices()
        all_voices = de_voices + en_voices or ["Anna", "Samantha"]
        rates = [150, 170, 190, 160, 180, 200]
        rate_factors = [0.9, 1.0, 1.1, 0.95, 1.05, 0.85]
        # Same neural voices as the positives — the model must reject the
        # phrase, not the voice
        piper_pool = download_piper_py_de_pool()
        for phrase in tqdm(phrases, desc="Confusable TTS"):
            for k in range(n_per_phrase):
                out_wav = out_dir / f"conf_{idx:04d}.wav"
                if not out_wav.exists():
                    pitch = pitch_shifts[k % len(pitch_shifts)]
                    try:
                        if piper_pool and k % 3 != 2:
                            onnx, spk = piper_pool[(idx * 7919) % len(piper_pool)]
                            piper_py_to_wav(phrase, onnx, spk,
                                            rate_factors[k % len(rate_factors)],
                                            pitch, out_wav, ffmpeg)
                        else:
                            voice = all_voices[k % len(all_voices)]
                            rate = rates[k % len(rates)]
                            say_to_wav(phrase, voice, rate, pitch, out_wav, ffmpeg)
                    except Exception:
                        pass
                idx += 1
    generated = len(list(out_dir.glob("*.wav")))
    print(f"  ✓ {generated} confusable WAVs ({len(phrases)} phrases × {n_per_phrase} variations)")


def _generate_mww_positive_features(pos_dir: Path, features_dir: Path,
                                     repetition: int = 2, eq_prob: float = 0.1,
                                     truncate_randomly: bool = False):
    """Convert WAV clips to Ragged MMap spectrograms.
    truncate_randomly=True picks a random 3.2s window from longer clips —
    use for negatives (sentences), NOT for end-aligned positives."""
    _mww_sys_path()
    from microwakeword.audio.clips import Clips
    from microwakeword.audio.augmentation import Augmentation
    from microwakeword.audio.spectrograms import SpectrogramGeneration
    from mmap_ninja.ragged import RaggedMmap

    clips = Clips(
        input_directory=str(pos_dir),
        file_pattern="*.wav",
        max_clip_duration_s=None,
        remove_silence=False,
        random_split_seed=42,
        split_count=0.1,
    )

    aug_kwargs: dict = dict(
        augmentation_duration_s=3.2,
        truncate_randomly=truncate_randomly,
        augmentation_probabilities={
            "SevenBandParametricEQ": eq_prob,
            "TanhDistortion":        0.1,
            "PitchShift":            0.1,
            "BandStopFilter":        0.1,
            "AddColorNoise":         0.1,
            "Gain":                  1.0,
        },
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )
    bg_paths = [
        str(DATA_DIR / name)
        for name in ["audioset_16k", "musan", "custom_background"]
        if (DATA_DIR / name).exists()
        and (
            list((DATA_DIR / name).rglob("*.wav"))
            or list((DATA_DIR / name).rglob("*.flac"))
        )
    ]
    rir_paths = [str(DATA_DIR / "mit_rirs")] if (DATA_DIR / "mit_rirs").exists() else []
    if bg_paths:
        aug_kwargs["augmentation_probabilities"]["AddBackgroundNoise"] = 0.75
        aug_kwargs["background_paths"] = bg_paths
        aug_kwargs["background_min_snr_db"] = -5
        aug_kwargs["background_max_snr_db"] = 10
    if rir_paths:
        aug_kwargs["augmentation_probabilities"]["RIR"] = 0.5
        aug_kwargs["impulse_paths"] = rir_paths

    augmenter = Augmentation(**aug_kwargs)

    for split, split_name, rep, slide in [
        ("training",   "train",      repetition, 10),
        ("validation", "validation", 1,           10),
        ("testing",    "test",       1,            1),
    ]:
        out_dir = features_dir / split
        mmap_dir = out_dir / "wakeword_mmap"
        # Check for dtype.ninja sentinel — directory alone may exist from a killed run
        if mmap_dir.exists() and (mmap_dir / "dtype.ninja").exists():
            print(f"  ✓ {split} spectrograms cached")
            continue
        if mmap_dir.exists():
            import shutil as _shutil
            _shutil.rmtree(str(mmap_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Generating {split} spectrograms …")
        sg = SpectrogramGeneration(clips=clips, augmenter=augmenter,
                                   slide_frames=slide, step_ms=10)
        RaggedMmap.from_generator(
            out_dir=str(mmap_dir),
            sample_generator=sg.spectrogram_generator(split=split_name, repeat=rep),
            batch_size=100,
            verbose=False,
        )
        print(f"  ✓ {split} done")


# ── Post-training model verification ──────────────────────────────────────────

def _verify_mww_model(tflite_path: Path, manifest_path: Path) -> bool:
    """Verify the trained model output — like a checksum check after download."""
    print("\n  ── Model Verification ────────────────────────────────────")
    checks: list[tuple[bool, str]] = []

    # 1) TFLite file exists and has reasonable size
    if tflite_path.exists():
        size_kb = tflite_path.stat().st_size // 1024
        ok = 20 <= size_kb <= 1000
        checks.append((ok, f"TFLite file: {size_kb} KB {'(OK)' if ok else '(WARN: unexpected size)'}"))
    else:
        checks.append((False, f"TFLite file: MISSING at {tflite_path}"))

    # 2) Manifest exists and has required fields
    required_keys = {"type", "wake_word", "version", "model", "micro"}
    if manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text())
            missing = required_keys - set(m.keys())
            micro_ok = isinstance(m.get("micro"), dict) and "probability_cutoff" in m.get("micro", {})
            ok = not missing and micro_ok
            detail = "valid" if ok else f"missing keys: {missing}"
            checks.append((ok, f"Manifest JSON: {detail}"))
        except Exception as e:
            checks.append((False, f"Manifest JSON: parse error — {e}"))
    else:
        checks.append((False, f"Manifest: MISSING at {manifest_path}"))

    # 3) Load TFLite model and run a dummy inference
    if tflite_path.exists():
        try:
            import tensorflow as tf
            interp = tf.lite.Interpreter(model_path=str(tflite_path))
            interp.allocate_tensors()
            inp = interp.get_input_details()
            out = interp.get_output_details()
            dummy = np.zeros(inp[0]["shape"], dtype=inp[0]["dtype"])
            interp.set_tensor(inp[0]["index"], dummy)
            interp.invoke()
            result = interp.get_tensor(out[0]["index"])
            raw = float(result.flat[0])
            # Dequantize if int8 output (quantized streaming model)
            qp = out[0].get("quantization", (1.0, 0))
            scale, zero_point = (qp[0], qp[1]) if qp[0] > 0 else (1.0, 0)
            prob = (raw - zero_point) * scale
            ok = 0.0 <= prob <= 1.0
            checks.append((ok, f"TFLite inference: p={prob:.4f} output={out[0]['shape']} {'(OK)' if ok else '(WARN: out of range)'}"))
        except Exception as e:
            checks.append((False, f"TFLite inference: failed — {e}"))

    # Print summary
    all_ok = all(ok for ok, _ in checks)
    for ok, msg in checks:
        print(f"  {'✓' if ok else '✗'} {msg}")
    if all_ok:
        print("  ── PASSED — model output looks healthy ──────────────────")
    else:
        print("  ── WARNING — some checks failed (see above) ─────────────")
    print()
    return all_ok


# ── microWakeWord training ─────────────────────────────────────────────────────

def _run_microwakeword_train(wake_word: str, model_dir: Path, n_samples: int, steps: int,
                             german_negatives: bool = True, german_clips: int = 6000):
    """Train a microWakeWord model (quantized int8 TFLite) for ESP32 on-device detection."""
    import shutil, subprocess as _sp

    if not _mww_repo_ok():
        print(f"""
╔══════════════════════════════════════════════════╗
║     microWakeWord: Repository Not Found          ║
╚══════════════════════════════════════════════════╝
  Expected: {MWW_REPO_DIR}

  Run ./setup.sh to clone it automatically, or:
    git clone https://github.com/kahrendt/microWakeWord {MWW_REPO_DIR}
""")
        sys.exit(1)

    try:
        import tensorflow as _tf  # noqa: F401
    except ImportError:
        print("""
╔══════════════════════════════════════════════════╗
║     microWakeWord: TensorFlow Required           ║
╚══════════════════════════════════════════════════╝
  Install with:
    pip install tensorflow
""")
        sys.exit(1)

    model_name = re.sub(r'[,!. ]+', '_', wake_word.lower()).strip('_')

    pos_dir          = model_dir / "positive_train"
    neg_user_dir     = model_dir / "negative_train"
    mww_work_dir     = model_dir / "mww"
    features_dir     = mww_work_dir / "positive_features"
    real_norm_dir    = mww_work_dir / "real_normalized"
    real_features_dir= mww_work_dir / "real_features"
    neg_features_dir = mww_work_dir / "negative_features"
    conf_wavs_dir    = mww_work_dir / "confusable_wavs"
    conf_features_dir= mww_work_dir / "confusable_features"
    adv_wavs_dir     = mww_work_dir / "adversarial_wavs"
    adv_features_dir = mww_work_dir / "adversarial_features"
    train_dir        = mww_work_dir / "trained_model"
    config_path      = mww_work_dir / "training_parameters.yaml"
    mww_work_dir.mkdir(parents=True, exist_ok=True)

    # ── Split real recordings from TTS positives ───────────────────────────────
    all_pos_wavs  = list(pos_dir.glob("*.wav"))
    real_wavs     = [w for w in all_pos_wavs if w.stem.startswith("real_")]
    tts_wavs      = [w for w in all_pos_wavs if not w.stem.startswith("real_")]
    has_real      = len(real_wavs) > 0

    # Build TTS-only symlink dir so positives_features only contains TTS
    tts_link_dir  = mww_work_dir / "tts_wavs"
    tts_link_dir.mkdir(exist_ok=True)
    for w in tts_wavs:
        link = tts_link_dir / w.name
        if not link.exists():
            link.symlink_to(w.resolve())

    has_custom_neg = (neg_user_dir.exists()
                      and any(neg_user_dir.glob("*.wav")))

    n_pos = len(all_pos_wavs)
    print(f"  {n_pos} positive clips ({len(real_wavs)} real, {len(tts_wavs)} TTS)")
    if has_custom_neg:
        print(f"  {len(list(neg_user_dir.glob('*.wav')))} custom negative clips")

    # ── Negative datasets (pre-generated spectrograms from HuggingFace) ────────
    print("\n  Checking negative datasets …")
    _download_mww_negatives()

    # ── German speech negatives (Common Voice — real German everyday speech) ───
    has_german = False
    if german_negatives:
        print("\n  Checking German speech negatives (Common Voice) …")
        has_german = _download_german_speech(n_train=german_clips)

    # ── TTS positive spectrograms (standard weight) ────────────────────────────
    print("\n  Generating TTS positive spectrograms …")
    _generate_mww_positive_features(tts_link_dir, features_dir, repetition=2, eq_prob=0.1)

    # ── Real recording spectrograms (higher weight, more repetition) ───────────
    ffmpeg = find_ffmpeg()
    if has_real:
        print("\n  LUFS-normalizing real recordings …")
        n_norm = _lufs_normalize_dir(pos_dir, real_norm_dir, ffmpeg, pattern="real_*.wav")
        print(f"  ✓ Normalized {n_norm} real recordings (-23 LUFS)")
        print("\n  Generating real recording spectrograms (repetition=5) …")
        _generate_mww_positive_features(real_norm_dir, real_features_dir,
                                         repetition=5, eq_prob=0.3)

    # ── Custom negative recordings (from recording UI → negative_train/) ───────
    if has_custom_neg:
        print("\n  Generating custom negative spectrograms …")
        _generate_mww_positive_features(neg_user_dir, neg_features_dir,
                                         repetition=2, eq_prob=0.1)

    # ── German speech negative spectrograms ────────────────────────────────────
    has_german_eval = False
    if has_german:
        print("\n  Generating German negative spectrograms …")
        # Invalidate cached features if the clip set grew (e.g. --german-clips raised)
        n_de = len(list(GERMAN_SPEECH_DIR.glob("*.wav")))
        de_count_file = GERMAN_FEATURES_DIR / "_clip_count.txt"
        if de_count_file.exists() and de_count_file.read_text() != str(n_de):
            shutil.rmtree(str(GERMAN_FEATURES_DIR))
        GERMAN_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        de_count_file.write_text(str(n_de))
        _generate_mww_positive_features(GERMAN_SPEECH_DIR, GERMAN_FEATURES_DIR,
                                         repetition=1, eq_prob=0.1,
                                         truncate_randomly=True)
        has_german_eval = _generate_german_eval_features()

    # ── Confusable phrase spectrograms ─────────────────────────────────────────
    print("\n  Generating confusable phrase TTS …")
    _generate_confusable_tts(MWW_CONFUSABLE_PHRASES, conf_wavs_dir, n_per_phrase=6)
    print("\n  Generating confusable spectrograms …")
    _generate_mww_positive_features(conf_wavs_dir, conf_features_dir,
                                     repetition=2, eq_prob=0.1)

    # ── Adversarial sentence spectrograms (same voices as positives) ───────────
    print("\n  Generating adversarial sentence TTS …")
    _generate_confusable_tts(MWW_ADVERSARIAL_SENTENCES, adv_wavs_dir, n_per_phrase=6)
    print("\n  Generating adversarial sentence spectrograms …")
    _generate_mww_positive_features(adv_wavs_dir, adv_features_dir,
                                     repetition=1, eq_prob=0.1,
                                     truncate_randomly=True)

    # ── Training config ────────────────────────────────────────────────────────
    # Multi-phase: decreasing LR + increasing negative penalty over 3 phases
    phase1 = int(steps * 0.55)
    phase2 = int(steps * 0.30)
    phase3 = steps - phase1 - phase2

    features_list = [
        # TTS positives (standard weight)
        {"features_dir": str(features_dir),
         "sampling_weight": 2.0, "penalty_weight": 1.0,
         "truth": True, "truncation_strategy": "truncate_start", "type": "mmap"},
    ]
    if has_real:
        # Real recordings: same weight as TTS block but more repetitions were applied
        features_list.append(
            {"features_dir": str(real_features_dir),
             "sampling_weight": 8.0, "penalty_weight": 2.0,
             "truth": True, "truncation_strategy": "truncate_start", "type": "mmap"}
        )
    features_list += [
        # HuggingFace negative datasets
        {"features_dir": str(MWW_NEG_DIR / "speech"),
         "sampling_weight": 10.0, "penalty_weight": 1.0,
         "truth": False, "truncation_strategy": "random", "type": "mmap"},
        {"features_dir": str(MWW_NEG_DIR / "dinner_party"),
         "sampling_weight": 10.0, "penalty_weight": 1.0,
         "truth": False, "truncation_strategy": "random", "type": "mmap"},
        {"features_dir": str(MWW_NEG_DIR / "no_speech"),
         "sampling_weight": 5.0, "penalty_weight": 1.0,
         "truth": False, "truncation_strategy": "random", "type": "mmap"},
        # Confusable hard negatives (phonetically similar phrases)
        {"features_dir": str(conf_features_dir),
         "sampling_weight": 8.0, "penalty_weight": 5.0,
         "truth": False, "truncation_strategy": "truncate_start", "type": "mmap"},
        # Adversarial German sentences, same TTS voices as positives — blocks
        # voice-identity/TTS-artifact shortcuts (arXiv:2201.00167)
        {"features_dir": str(adv_features_dir),
         "sampling_weight": 6.0, "penalty_weight": 2.0,
         "truth": False, "truncation_strategy": "truncate_start", "type": "mmap"},
        # Ambient eval set (sampling_weight=0 → eval only, not training)
        {"features_dir": str(MWW_NEG_DIR / "dinner_party_eval"),
         "sampling_weight": 0.0, "penalty_weight": 1.0,
         "truth": False, "truncation_strategy": "split", "type": "mmap"},
    ]
    if has_german:
        # Real German everyday speech (Common Voice) — the HF negatives above are
        # English-only; this is what actually causes household false triggers
        features_list.append(
            {"features_dir": str(GERMAN_FEATURES_DIR),
             "sampling_weight": 12.0, "penalty_weight": 2.0,
             "truth": False, "truncation_strategy": "truncate_start", "type": "mmap"})
    if has_german_eval:
        # German ambient eval (sampling_weight=0) — makes reported FA/h include German speech
        features_list.append(
            {"features_dir": str(GERMAN_EVAL_FEATURES_DIR),
             "sampling_weight": 0.0, "penalty_weight": 1.0,
             "truth": False, "truncation_strategy": "split", "type": "mmap"})
    if has_custom_neg:
        # User-recorded negative phrases (highest penalty — exact confusables for this household)
        features_list.insert(-1,
            {"features_dir": str(neg_features_dir),
             "sampling_weight": 15.0, "penalty_weight": 5.0,
             "truth": False, "truncation_strategy": "truncate_start", "type": "mmap"}
        )

    config = {
        "window_step_ms": 10,
        "train_dir": str(train_dir),
        "features": features_list,
        # 3-phase schedule: broad learning → refinement → fine-tuning
        "training_steps":        [phase1,  phase2,  phase3],
        "positive_class_weight": [1,       1,       1],
        "negative_class_weight": [20,      30,      50],
        "learning_rates":        [0.001,   0.0005,  0.0001],
        "batch_size":            128,
        # SpecAugment: time + frequency masking improves generalization
        "time_mask_max_size":    [5,       5,       5],
        "time_mask_count":       [1,       2,       2],
        "freq_mask_max_size":    [5,       5,       5],
        "freq_mask_count":       [1,       2,       2],
        "eval_step_interval":    500,
        # ~2s window: slow/long "Hey Dobbi" variants (up to 1.75s) must fit
        # completely after the ~0.2s end-jitter — at 1500ms the "Hey" of any
        # clip >1.3s was silently truncated away during training.
        # MUST yield a spectrogram_length divisible by the model stride (3):
        # (1 + (16*ms - 480)//480 + slices_dropped) % 3 == 0 — else the INT8
        # calibration asserts. 2040 → length 222 ✓ (2000 → 220 ✗)
        "clip_duration_ms":      2040,
        "target_minimization":   0.9,
        "minimization_metric":   None,
        "maximization_metric":   "average_viable_recall",
    }
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    print(f"\n  ✓ Config: {config_path.name} ({steps} steps in 3 phases: {phase1}+{phase2}+{phase3})")

    # ── Run model_train_eval as subprocess ─────────────────────────────────────
    env = {**os.environ, "PYTHONPATH": str(MWW_REPO_DIR)}
    cmd = [
        sys.executable, "-m", "microwakeword.model_train_eval",
        "--training_config", str(config_path),
        "--train", "1",
        "--restore_checkpoint", "1",
        "--test_tf_nonstreaming", "0",
        "--test_tflite_nonstreaming", "0",
        "--test_tflite_nonstreaming_quantized", "0",
        "--test_tflite_streaming", "0",
        "--test_tflite_streaming_quantized", "1",
        "--use_weights", "best_weights",
        "mixednet",
        "--pointwise_filters", "64,64,64,64",
        "--repeat_in_block", "1,1,1,1",
        "--mixconv_kernel_sizes", "[5],[7,11],[9,15],[23]",
        "--residual_connection", "0,0,0,0",
        "--first_conv_filters", "32",
        "--first_conv_kernel_size", "5",
        "--stride", "3",
    ]
    print(f"\n  Running model_train_eval ({steps} steps) …")
    result = _sp.run(cmd, env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)

    # ── Copy TFLite output and generate ESPHome manifest ──────────────────────
    tflite_src = (train_dir / "tflite_stream_state_internal_quant"
                  / "stream_state_internal_quant.tflite")
    tflite_dst = OUTPUT_DIR / f"{model_name}.tflite"

    if tflite_src.exists():
        shutil.copy2(str(tflite_src), str(tflite_dst))

    # Read HA URL from settings.json (set once via web UI → Settings gear)
    ha_url = "http://YOUR_HA_IP:8123"
    settings_path = BASE_DIR / "settings.json"
    if settings_path.exists():
        try:
            _s = json.loads(settings_path.read_text())
            if _s.get("haUrl"):
                ha_url = _s["haUrl"].rstrip("/")
        except Exception:
            pass

    # ── Auto-select probability_cutoff from the streaming ROC ─────────────────
    # Pick the LOWEST cutoff whose false-accepts/hour ≤ 0.1 (best recall at
    # near-zero false alarms). Falls back to 0.97 if the ROC file is missing.
    cutoff = 0.97
    roc_path = (train_dir / "tflite_stream_state_internal_quant"
                / "tflite_streaming_roc.txt")
    try:
        candidates = []
        for line in roc_path.read_text().splitlines():
            m = re.match(r"Cutoff ([\d.]+): frr=([\d.]+); faph=([\d.]+)", line.strip())
            if m:
                candidates.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
        viable = [c for c in candidates if c[2] <= 0.1]
        if viable:
            cutoff = min(viable, key=lambda c: (c[0], c[1]))[0]
            frr = min(viable, key=lambda c: (c[0], c[1]))[1]
            print(f"\n  Auto-selected probability_cutoff={cutoff} (FRR {frr*100:.1f}%, ≤0.1 FA/h)")
        elif candidates:
            cutoff = max(c[0] for c in candidates)
            print(f"\n  ⚠ No cutoff reaches ≤0.1 FA/h — using strictest ({cutoff})")
    except OSError:
        print("\n  ⚠ ROC file missing — using default probability_cutoff=0.97")

    # ESPHome micro_wake_word v2 manifest format (https://esphome.io/components/micro_wake_word.html)
    # model is a relative path — resolved via urljoin against the manifest URL at compile time
    manifest = {
        "type":              "micro",
        "wake_word":         wake_word,
        "author":            "Speaky",
        "version":           2,
        "model":             tflite_dst.name,
        "trained_languages": ["de"],
        "micro": {
            # Rolling-average threshold, auto-selected from the streaming ROC
            "probability_cutoff":      cutoff,
            "feature_step_size":       10,
            # 10 windows × 30ms = 300ms averaging — robust against brief noise triggers
            "sliding_window_size":     10,
            # Our 64-filter mixednet needs more arena than the official 26 KB models
            "tensor_arena_size":       45000,
            "minimum_esphome_version": "2024.7",
        },
    }
    manifest_path = OUTPUT_DIR / f"{model_name}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Post-training model verification ─────────────────────────────────────
    _verify_mww_model(tflite_dst, manifest_path)

    url_hint = "" if ha_url != "http://YOUR_HA_IP:8123" else \
        "\n  ⚠  HA-URL nicht konfiguriert — bitte im Web-UI (Zahnrad oben rechts) eintragen!\n"
    size_kb = tflite_dst.stat().st_size // 1024 if tflite_dst.exists() else "?"
    model_id = model_name  # e.g. "hey_dobbi"
    print(f"""
╔══════════════════════════════════════════════════╗
║        microWakeWord Training Complete!          ║
╚══════════════════════════════════════════════════╝
  Model    : {tflite_dst}
  Size     : {size_kb} KB
  Manifest : {manifest_path}
  HA URL   : {ha_url}{url_hint}

  ── Install steps ──────────────────────────────────
  1. Copy to HA:  /config/www/{tflite_dst.name}
                  /config/www/{manifest_path.name}

  2. Replace micro_wake_word: block in ESPHome YAML:

micro_wake_word:
  id: mww
  microphone:
    microphone: i2s_mics
    channels: 1
  stop_after_detection: false
  vad:
    model: github://esphome/micro-wake-word-models/models/v2/vad.json@main
  models:
    - model: {ha_url}/local/{model_name}_manifest.json
      id: {model_id}
    - model: github://esphome/micro-wake-word-models/models/v2/stop.json@main
      id: stop
      internal: true

  3. Flash the ESPHome device and test.
     If false positives occur: raise probability_cutoff in manifest toward 0.97
     If wake word is missed:   lower probability_cutoff toward 0.90
""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Wake word trainer for Home Assistant (macOS ARM / Apple Silicon)"
    )
    parser.add_argument("wake_word", nargs="?", default=None,
                        help='Wake word, e.g. "Hey Dobbi"')
    parser.add_argument("--samples", type=int, default=500,
                        help="TTS samples to generate (default: 500)")
    parser.add_argument("--steps", type=int, default=3000,
                        help="Training steps (default: 3000)")
    parser.add_argument("--full", action="store_true",
                        help="Production mode: 2000 samples, 25k steps, +11 GB downloads")
    parser.add_argument("--prefetch", action="store_true",
                        help="Download all training data (~13 GB) without training")
    parser.add_argument("--platform", type=str, default="openWakeWord",
                        choices=["openWakeWord", "microWakeWord", "both"],
                        help="Target platform: openWakeWord (Wyoming/HA), microWakeWord (ESP32), or both")
    parser.add_argument("--no-german-negatives", action="store_true",
                        help="Skip downloading German Common Voice speech as hard negatives (microWakeWord)")
    parser.add_argument("--german-clips", type=int, default=6000,
                        help="Number of German Common Voice clips to use as negatives (default 6000, ~10h)")
    parser.add_argument("--lang", type=str, default=os.environ.get("SPEAKY_LANG", "de"),
                        choices=["de", "en"],
                        help="TTS language for sample generation (default: de or $SPEAKY_LANG)")
    args = parser.parse_args()

    if args.prefetch:
        print("""
╔══════════════════════════════════════════════════╗
║         Pre-fetching All Training Data           ║
╚══════════════════════════════════════════════════╝
  Downloads (cached in ./data/ — only once):
    • MIT Room Impulse Responses      ~50 MB
    • AudioSet background clips       ~500 clips
    • MUSAN music/noise               ~200 clips
    • Validation features             ~400 MB
    • ACAV100M negative features      ~11 GB
""")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print("[1/2] Downloading background / augmentation data...")
        download_background_data(full_mode=True)
        print("\n[2/2] Downloading negative feature data...")
        download_feature_data(full_mode=True)
        print("""
╔══════════════════════════════════════════════════╗
║              All data ready!                     ║
╚══════════════════════════════════════════════════╝
  Run training with:
    python train.py "Hey Dobbi" --full
    make train WORD="Hey Dobbi" FULL=1
""")
        return

    if not args.wake_word:
        parser.error("wake_word is required (or use --prefetch to just download data)")

    n_samples = (2000 if args.full else args.samples) if args.samples == 500 else args.samples
    steps = (25000 if args.full else args.steps) if args.steps == 3000 else args.steps

    device = get_device()

    tts_backend = "Piper TTS (Linux/Docker)" if _IS_LINUX else "macOS say"
    platform_label = (
        "ESP32 on-device (microWakeWord)" if args.platform == "microWakeWord"
        else "ESP32 + Wyoming (both)" if args.platform == "both"
        else "Wyoming / HA server (openWakeWord)"
    )
    print(f"""
╔══════════════════════════════════════════════════╗
║           Speaky — Wake Word Trainer           ║
╚══════════════════════════════════════════════════╝
  Wake word : "{args.wake_word}"
  Platform  : {platform_label}
  Samples   : {n_samples}
  Steps     : {steps}
  Device    : {device}
  TTS       : {tts_backend}
  Mode      : {"FULL (production)" if args.full else "QUICK (test)"}
""")

    model_name = re.sub(r'[,!. ]+', '_', args.wake_word.lower()).strip('_')
    model_dir = OUTPUT_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── microWakeWord path ─────────────────────────────────────────────────────
    if args.platform == "microWakeWord":
        tts_desc = "Piper TTS voices" if _IS_LINUX else "macOS voices"
        print(f"\n[1/3] Generating {n_samples} TTS samples with {tts_desc} ({args.lang})...")
        generate_samples(args.wake_word, model_dir, n_samples, lang=args.lang)

        print("\n[2/3] Downloading background audio for augmentation...")
        download_background_data(args.full)

        print(f"\n[3/3] Training microWakeWord model ({steps} steps)...")
        _run_microwakeword_train(args.wake_word, model_dir, n_samples, steps,
                                 german_negatives=not args.no_german_negatives,
                                 german_clips=args.german_clips)
        return

    # ── both platforms path ────────────────────────────────────────────────────
    if args.platform == "both":
        tts_desc = "Piper TTS voices" if _IS_LINUX else "macOS voices"
        print(f"\n[1/6] Generating {n_samples} TTS samples with {tts_desc} ({args.lang})...")
        generate_samples(args.wake_word, model_dir, n_samples, lang=args.lang)

        print("\n[2/6] Downloading background audio for augmentation...")
        download_background_data(args.full)

        print(f"\n[3/6] Training microWakeWord model ({steps} steps)...")
        _run_microwakeword_train(args.wake_word, model_dir, n_samples, steps,
                                 german_negatives=not args.no_german_negatives,
                                 german_clips=args.german_clips)

        if not _IS_LINUX:
            print("\n[4/6] Patching openWakeWord for Apple Silicon MPS...")
            patch_openwakeword_mps()
        else:
            print("\n[4/6] Linux/Docker mode — skipping MPS patch")

        print("\n[5/6] Downloading negative training features...")
        download_feature_data(args.full)

        negative_features_path = str(
            DATA_DIR / ("openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
                        if args.full else "validation_set_features.npy")
        )
        config_path, model_name_oww = make_config(
            args.wake_word, negative_features_path, args.full, steps, n_samples
        )
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print(f"\n[6/6] Training openWakeWord model ({steps} steps on {device})...")
        _run_augment_and_train(config_path, model_name_oww, model_dir, steps, args.wake_word)
        return

    # ── openWakeWord path ──────────────────────────────────────────────────────
    if not _IS_LINUX:
        print("[1/5] Patching openWakeWord for Apple Silicon MPS...")
        patch_openwakeword_mps()
    else:
        print("[1/5] Linux/Docker mode — skipping MPS patch")

    # Start downloads in background threads so they run in parallel with TTS
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _bg_errors: list[Exception] = []

    def _bg_download():
        try:
            download_background_data(args.full)
        except Exception as e:
            _bg_errors.append(e)

    def _feat_download():
        try:
            download_feature_data(args.full)
        except Exception as e:
            _bg_errors.append(e)

    bg_thread = threading.Thread(target=_bg_download, daemon=True)
    feat_thread = threading.Thread(target=_feat_download, daemon=True)
    bg_thread.start()
    feat_thread.start()

    tts_desc = "Piper TTS voices" if _IS_LINUX else "macOS voices"
    print(f"\n[2/5] Generating {n_samples} TTS samples with {tts_desc} ({args.lang})...")
    print("      (downloads running in background — will wait before training)")
    generate_samples(args.wake_word, model_dir, n_samples, lang=args.lang)

    print("\n[3/5] Waiting for background / augmentation data...")
    bg_thread.join()

    print("\n[4/5] Waiting for negative training features...")
    feat_thread.join()

    if _bg_errors:
        print(f"  ⚠ Download warning: {_bg_errors[0]}")

    negative_features_path = str(
        DATA_DIR / ("openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
                    if args.full else "validation_set_features.npy")
    )

    config_path, model_name = make_config(
        args.wake_word, negative_features_path, args.full, steps, n_samples
    )
    print(f"\n  ✓ Config: {config_path}")

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    print(f"\n[5/5] Training model ({steps} steps on {device})...")
    _run_augment_and_train(config_path, model_name, model_dir, steps, args.wake_word)

    onnx_files = list(OUTPUT_DIR.glob(f"**/{model_name}*.onnx"))
    model_file = onnx_files[0] if onnx_files else model_dir

    print(f"""
╔══════════════════════════════════════════════════╗
║                 Training Complete!               ║
╚══════════════════════════════════════════════════╝
  Model : {model_file}

  To install in Home Assistant:
  → Copy .onnx file to /share/openwakeword/ on your HA host
  → Restart the Wyoming openWakeWord add-on
  → Settings → Voice Assistants → Wake word → select "{args.wake_word}"
""")


if __name__ == "__main__":
    try:
        main()
        # Signal the web UI that training completed successfully.
        # Written directly so Node.js close-handler loss (e.g. dev-server restart)
        # does not leave the run stuck in "running" state.
        print("\n__DONE__", flush=True)
    except SystemExit as e:
        if e.code != 0:
            print("\n__FAILED__", flush=True)
        raise
    except Exception:
        print("\n__FAILED__", flush=True)
        raise
