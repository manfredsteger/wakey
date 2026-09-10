# ── Stage 1: Python trainer base ──────────────────────────────────────────────
FROM python:3.11-slim AS trainer

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=0

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        curl \
        ca-certificates \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Piper TTS binary (auto-detects aarch64 / x86_64)
RUN ARCH=$(uname -m) && \
    mkdir -p /usr/local/piper && \
    curl -fsSL \
        "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_${ARCH}.tar.gz" \
        | tar -xz -C /usr/local/ && \
    test -f /usr/local/piper/piper && \
    echo "Piper installed: $(/usr/local/piper/piper --version 2>&1 || echo ok)"

# TensorFlow for microWakeWord training
RUN pip install --no-cache-dir tensorflow

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone microWakeWord and install from source (pip wheel v0.1.0 is incomplete)
# tbm_utils installed without deps: its pendulum dep requires Rust and fails on Python 3.13
RUN git clone --depth 1 https://github.com/kahrendt/microWakeWord.git microWakeWord && \
    pip install --no-cache-dir --no-deps -e microWakeWord && \
    pip install --no-cache-dir --no-deps tbm_utils

# Clone openWakeWord
RUN git clone --depth 1 https://github.com/dscripka/openWakeWord.git openWakeWord && \
    pip install --no-cache-dir -e openWakeWord

# Download openWakeWord base models into the package resources dir
RUN python - <<'EOF'
from huggingface_hub import hf_hub_download
import shutil, pathlib
dst = pathlib.Path("openWakeWord/openwakeword/resources/models")
dst.mkdir(parents=True, exist_ok=True)
for name in ["embedding_model.onnx", "melspectrogram.onnx"]:
    src = hf_hub_download("davidscripka/openwakeword", name, repo_type="model")
    shutil.copy(src, dst / name)
    print(f"  {name} -> {dst / name}")
EOF

# Create piper-sample-generator stub (openWakeWord imports it at module level)
COPY piper-sample-generator/generate_samples.py piper-sample-generator/

# Apply patches
COPY patches/ patches/

# 1. torchaudio shim → openWakeWord/openwakeword/data.py
RUN python patches/patch_oww_data.py

# 2. Fix acoustics/directivity.py (scipy 1.15 removed sph_harm)
RUN python - <<'EOF'
import site, pathlib, textwrap
sp = pathlib.Path(site.getsitepackages()[0])
f = sp / "acoustics/directivity.py"
if not f.exists():
    print("  acoustics not found — skip"); exit(0)
t = f.read_text()
old = "from scipy.special import sph_harm"
new = textwrap.dedent("""\
    try:
        from scipy.special import sph_harm
    except ImportError:
        from scipy.special import sph_harm_y as _sph_harm_y
        def sph_harm(m, n, theta, phi):
            return _sph_harm_y(n, m, theta, phi)""")
if old in t and "try:" not in t:
    f.write_text(t.replace(old, new))
    print("  Patched acoustics/directivity.py")
else:
    print("  acoustics/directivity.py already patched or different version")
EOF

# 3. Fix pronouncing/__init__.py (dead pkg_resources import)
RUN python - <<'EOF'
import site, pathlib
sp = pathlib.Path(site.getsitepackages()[0])
f = sp / "pronouncing/__init__.py"
if not f.exists():
    print("  pronouncing not found — skip"); exit(0)
t = f.read_text()
bad = "from pkg_resources import resource_stream\n"
if bad in t:
    f.write_text(t.replace(bad, ""))
    print("  Patched pronouncing/__init__.py")
else:
    print("  pronouncing/__init__.py already clean")
EOF

# Copy trainer script
COPY train.py record.py ./

# Volumes for persistent data and output models
VOLUME ["/app/data", "/app/output", "/app/piper-voices"]

ENTRYPOINT ["python", "train.py"]
CMD ["--help"]


# ── Stage 2: Web UI (Python trainer + Node.js) ─────────────────────────────────
FROM trainer AS web

# Install Node.js 20 LTS
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install web dependencies
COPY web/package.json /app/web/
RUN cd /app/web && npm install --prefer-offline

# Copy web source and build
COPY web/ /app/web/
RUN cd /app/web && \
    npx prisma generate && \
    npm run build

EXPOSE 3000

CMD ["sh", "-c", "cd /app/web && npx prisma db push --skip-generate 2>/dev/null; npm start"]
