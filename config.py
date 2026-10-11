from pathlib import Path

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "FacialTracking_Actors_01-24"
PARQUET_PATH = ""
SEED = 42
TEST_ACTORS = ["01", "16", "17", "24"]
EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}