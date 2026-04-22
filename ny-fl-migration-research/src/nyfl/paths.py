"""Where the big folders live so we’re not hard-coding strings all over the repo."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SAMPLE = ROOT / "data" / "sample"
OUTPUTS = ROOT / "outputs" / "figures"
