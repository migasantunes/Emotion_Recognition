from pathlib import Path

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "FacialTracking_Actors_01-24"
PARQUET_PATH = ROOT / "data" / "Parquet" / "frame_table.parquet"
SEED = 42
TEST_ACTORS = [1, 16, 17, 24]
EMOTIONS = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised"
}