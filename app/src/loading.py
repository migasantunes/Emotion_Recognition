import config as cfg
import pandas as pd
import numpy as np

def parse_filename(filename): # example: 01-01-06-02-01-02-12.csv
    fields = np.array(filename.split("-")).astype(int)

    sex = "M" if fields[6] % 2 == 1 else "F"
    is_test = 1 if fields[6] in cfg.TEST_ACTORS else 0

    return {"video_id": filename,
            "actor": fields[6],
            "sex": sex,
            "channel": fields[1],
            "emotion": fields[2],
            "intensity": fields[3],
            "statement": fields[4],
            "repetition": fields[5],
            "is_test": is_test}

def read_file(filepath):
    filename = filepath.stem

    column_names = ["frame", "timestamp", "confidence"] + [f"x_{i}" for i in range(68)] + [f"y_{i}" for i in range(68)]

    file_df = pd.read_csv(filepath, usecols=column_names)
    
    fields = parse_filename(filename)
    fields.pop("channel")
    
    fields_df = pd.DataFrame(fields, index=file_df.index)

    return pd.concat([fields_df, file_df[column_names]], axis=1) # Just to reorder the columns, to match the order in the brief

def load_frames(include_tests=False, force=False): 
    cfg.PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)

    if force or not cfg.PARQUET_PATH.exists():
        files = list()
        for file in sorted(cfg.DATA_PATH.glob("*.csv")):
            fields = parse_filename(file.stem)

            if fields["channel"] == 1:
                files.append(read_file(file))

        df = pd.concat(files, ignore_index=True).round({"timestamp": 3})
        df.to_parquet(cfg.PARQUET_PATH, index=False)

    else:
        df = pd.read_parquet(cfg.PARQUET_PATH)

    if not include_tests:
        df = df[df["is_test"] == 0]

    df = df.reset_index(drop=True)

    return df

def video_table(frames):
    return frames.groupby("video_id").agg(
        actor = ("actor", "first"),
        sex = ("sex", "first"),
        emotion = ("emotion", "first"),
        intensity = ("intensity", "first"),
        statement = ("statement", "first"),
        repetition = ("repetition", "first"),
        n_frames = ("frame", "size")
    ).reset_index()