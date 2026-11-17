"""Settings. API keys live in the file named  .env  (copy .env.example to .env)."""
import os
from pathlib import Path

ROOT = Path(__file__).parent
ENV_FILE = ROOT / ".env"


def _load_env():
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()


def get_key(name):
    v = os.environ.get(name, "").strip()
    return v if v and not v.startswith("PASTE_") else None


DB_PATH = ROOT / "football.db"
