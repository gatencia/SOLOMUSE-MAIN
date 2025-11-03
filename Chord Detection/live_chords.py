#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Live chord / key / tempo listener for SoloMuse.

- Reads loopback (e.g., BlackHole) or mic
- Stabilizes tempo (median + x2/÷2 snap), key, and chord labels
- Uses Vamp Chordino (nnls-chroma:chordino)
- Optional POST to a local server
- Loads configuration from a parent-folder .env (SOLOMUSE-MAIN/.env)

Create SOLOMUSE-MAIN/.env from the provided .env.example.
"""

import os
import sys
import ctypes
from pathlib import Path
from collections import deque, Counter

# -------------------------------
# Load .env from parent directory
# -------------------------------
ROOT = Path(__file__).resolve().parents[1]  # SOLOMUSE-MAIN
ENV_PATH = ROOT / ".env"

def _parse_bool(s, default=False):
    if s is None:
        return default
    return str(s).strip().lower() in {"1", "true", "t", "yes", "y", "on"}

def _load_env():
    # Prefer python-dotenv if installed; otherwise, parse a simple KEY=VALUE file.
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(ENV_PATH)  # does nothing if file is missing
    except Exception:
        if ENV_PATH.exists():
            for line in ENV_PATH.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

_load_env()

# -------------------------------
# Read config (with sane defaults)
# -------------------------------
INPUT_DEVICE_NAME = os.environ.get("INPUT_DEVICE_NAME", "BlackHole")
SAMPLE_RATE       = int(float(os.environ.get("SAMPLE_RATE", "44100")))
DURATION          = float(os.environ.get("DURATION", "2.0"))
N_FFT             = int(float(os.environ.get("N_FFT", "2048")))
HOP               = int(float(os.environ.get("HOP", "512")))
VAMP_PLUGIN_DIR   = os.path.expanduser(os.environ.get("VAMP_PLUGIN_DIR", "")) or None
MANUAL_DYLIB      = os.path.expanduser(os.environ.get("MANUAL_CHORDINO_DYLIB", "")) or None
TEMPO_HISTORY     = int(float(os.environ.get("TEMPO_HISTORY", "8")))
KEY_HISTORY       = int(float(os.environ.get("KEY_HISTORY", "12")))
CHORD_HISTORY     = int(float(os.environ.get("CHORD_HISTORY", "8")))
POST_TO_SERVER    = _parse_bool(os.environ.get("POST_TO_SERVER", "false"))
SERVER_URL        = os.environ.get("SERVER_URL", "") or None
PRINT_PLUGIN_LIST = _parse_bool(os.environ.get("PRINT_PLUGIN_LIST", "true"))

# -------------------------------
# Resolve and export VAMP paths
# -------------------------------
def resolve_vamp_dir():
    # 1) .env override
    if VAMP_PLUGIN_DIR and os.path.isdir(VAMP_PLUGIN_DIR):
        return VAMP_PLUGIN_DIR
    # 2) Env var (user shell)
    env = os.environ.get("VAMP_PATH")
    if env and os.path.isdir(env):
        return env
    # 3) Common macOS locations
    user_dir = os.path.expanduser("~/Library/Audio/Plug-Ins/Vamp")
    sys_dir  = "/Library/Audio/Plug-Ins/Vamp"
    return user_dir if os.path.isdir(user_dir) else sys_dir

VAMP_DIR = resolve_vamp_dir()
os.environ["VAMP_PATH"] = VAMP_DIR
print(f"🔎 VAMP_PATH = {VAMP_DIR}")

# Try to preload the Chordino dylib to help some hosts discover it
def preload_chordino():
    dylib = MANUAL_DYLIB or os.path.join(VAMP_DIR, "nnls-chroma.dylib")
    if os.path.exists(dylib):
        try:
            ctypes.cdll.LoadLibrary(dylib)
            print(f"✅ Preloaded: {dylib}")
        except OSError as e:
            print(f"⚠️  Failed to preload '{dylib}': {e}")

preload_chordino()

# -------------------------------
# Imports that depend on VAMP_PATH
# -------------------------------
import vamp  # noqa: E402
import sounddevice as sd  # noqa: E402
import numpy as np  # noqa: E402
import tempfile  # noqa: E402
import soundfile as sf  # noqa: E402
import librosa  # noqa: E402
try:
    import requests  # noqa: E402
except Exception:
    requests = None

from chord_extractor.extractors import Chordino  # noqa: E402

# Show plugins for sanity
try:
    plugins = vamp.list_plugins()
    if PRINT_PLUGIN_LIST:
        print(f"✅ Vamp plugins detected: {plugins}")
    if "nnls-chroma:chordino" not in plugins:
        print("⚠️  'nnls-chroma:chordino' not listed. Ensure the Vamp Plugin Pack is installed in VAMP_PATH.")
except Exception as e:
    print(f"⚠️  Could not list Vamp plugins: {e}")

# -------------------------------
# Audio device selection (prefer loopback)
# -------------------------------
def pick_input(name_like=None):
    """Return device index for an input device that contains `name_like` (case-insensitive).
       If not found, return default input device."""
    try:
        devs = sd.query_devices()
    except Exception:
        return None
    if name_like:
        nl = name_like.lower()
        for i, d in enumerate(devs):
            if d.get("max_input_channels", 0) > 0 and nl in d["name"].lower():
                return i
    # default input
    default = sd.default.device
    if isinstance(default, (list, tuple)) and len(default) > 0:
        return default[0]
    return None

INPUT_DEVICE = pick_input(INPUT_DEVICE_NAME)

if INPUT_DEVICE is not None:
    try:
        sd.check_input_settings(device=INPUT_DEVICE, samplerate=SAMPLE_RATE, channels=1)
    except Exception as e:
        print(f"⚠️  Input device check failed: {e}. Falling back to system default.")
        INPUT_DEVICE = None

# -------------------------------
# Global analyzers & history
# -------------------------------
TEMPO_HIST = deque(maxlen=TEMPO_HISTORY)
KEY_HIST   = deque(maxlen=KEY_HISTORY)
CHORD_HIST = deque(maxlen=CHORD_HISTORY)

chordino = Chordino(roll_on=1)  # uses nnls-chroma:chordino
PLUGIN_KEY = "nnls-chroma:chordino"

# -------------------------------
# Utilities
# -------------------------------
def ensure_len(y, min_len):
    if len(y) < min_len:
        return np.pad(y, (0, min_len - len(y)))
    return y

def stable_tempo(new_t):
    new_t = float(np.squeeze(new_t))
    if TEMPO_HIST:
        last = float(np.median(TEMPO_HIST))
        # Snap 2x or 1/2x if very close
        if new_t > 1.8 * last and abs(new_t / 2 - last) / max(last, 1e-6) < 0.12:
            new_t = new_t / 2
        elif new_t < 0.55 * last and abs(new_t * 2 - last) / max(last, 1e-6) < 0.12:
            new_t = new_t * 2
    TEMPO_HIST.append(new_t)
    return float(np.median(TEMPO_HIST))

def get_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    major_template = np.roll([1,0,0,1,0,0,0,1,0,0,0,0], 0)
    minor_template = np.roll([1,0,1,0,0,1,0,0,1,0,0,0], 0)
    major_scores = [np.dot(np.roll(major_template, k), chroma_mean) for k in range(12)]
    minor_scores = [np.dot(np.roll(minor_template, k), chroma_mean) for k in range(12)]
    pitch_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    key_index = int(np.argmax(major_scores + minor_scores))
    key_name = pitch_names[key_index % 12]
    scale = 'major' if key_index < 12 else 'minor'
    return key_name, scale

def stable_key(k, s):
    KEY_HIST.append((k, s))
    return Counter(KEY_HIST).most_common(1)[0][0]  # (key, scale)

def stable_chord(label):
    CHORD_HIST.append(label)
    counts = Counter([c for c in CHORD_HIST if c != "N"])
    if counts:
        return counts.most_common(1)[0][0]
    return "N"

def record_chunk(seconds):
    print(f"\n🎸 Listening for {seconds} seconds...")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=INPUT_DEVICE,
    )
    sd.wait()
    return audio.flatten()

# -------------------------------
# Main analysis
# -------------------------------
def analyze(audio):
    """Analyze one chunk for tempo, key, mood, and chord progression."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, audio, SAMPLE_RATE)
        y, sr = librosa.load(tmp.name, sr=None, mono=True)
        y = ensure_len(y, N_FFT)

        # Tempo (smoothed + double/half fix)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP)
        tempo = stable_tempo(tempo)

        # Key (smoothed)
        key, scale = get_key(y, sr)
        key, scale = stable_key(key, scale)

        # Mood features
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)))
        rms = float(np.mean(librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)))

        # Chords (with debouncing) via Chordino
        try:
            chords = chordino.extract(tmp.name)
        except Exception:
            # Fallback via vamp host directly
            try:
                data, rate = sf.read(tmp.name)
                res = vamp.collect(data, rate, PLUGIN_KEY)
                chords = res.get("list", [])
            except Exception:
                chords = []

        labels = [getattr(c, "chord", None) or c.get("label", "N") for c in (chords or [])]
        labels = [l for l in labels if l]
        last_meaningful = next((l for l in reversed(labels) if l != "N"), "N") if labels else "N"
        last_meaningful = stable_chord(last_meaningful)

        # Console HUD
        sys.stdout.write(
            f"\r🎶 Now playing: {last_meaningful:<7} | Key: {key:<2} {scale:<5} | "
            f"Tempo: {tempo:5.1f} BPM | Centroid: {int(centroid):>5} | RMS: {rms:.3f} | ZCR: {zcr:.3f}"
        )
        sys.stdout.flush()

        # Optional POST
        if POST_TO_SERVER and requests and SERVER_URL:
            payload = {
                "tempo": float(tempo),
                "key": key,
                "scale": scale,
                "mood": {"centroid": float(centroid), "rms": float(rms), "zcr": float(zcr)},
                "current_chord": last_meaningful,
            }
            try:
                requests.post(SERVER_URL, json=payload, timeout=0.2)
            except Exception:
                pass

# -------------------------------
# Entrypoint
# -------------------------------
def main():
    print("🎧 Real-time chord listener started (Ctrl+C to stop).")
    try:
        while True:
            chunk = record_chunk(DURATION)
            analyze(chunk)
    except KeyboardInterrupt:
        print("\n👋 Stopped listening.")

if __name__ == "__main__":
    main()