# IML Project Plan — Emotion Recognition

Oct 9, 2026 · @José Miguel Luís Antunes

## Overview

The project is split into two parallel tracks that meet at three sync points, with everything coded by 21 Oct and submitted by 23 Oct, one day before the 24 Oct 23:59 deadline.

- **Track A — the pipeline (Miguel):** the critical path. Data loading, normalisation, the 30 base features, the evaluation protocol and choice of C, the test, and the app's `classifier.py`. Everything downstream depends on it, so it gets the person who can move fastest on code.
- **Track B — the analysis (partner):** the exploration, data-quality study, own features, PCA and feature selection, and the deployment analysis. Mostly independent once the frame table and the normalisation function exist.

**Two golden rules — breaking either voids questions, not just points:**

1. **Test actors 01, 16, 17, 24 are invisible until 5.7.** No plots, no stats, no normalisation checks, nothing. Filter them out at load time in one place and never touch them.
2. **Nothing is fitted outside a Pipeline on training actors.** Scaler, PCA, SelectKBest, mRMR and C all fit inside each fold. Groups are always the actor.

**The defence (26 Oct) is individual and multiplies the project mark.** A 0 % defence zeroes the project. Splitting the work is fine; splitting the understanding is not. Every merge is reviewed by the other person, and the last two days are for learning each other's half.

## Repo structure and shared code

All reusable logic lives in `src/` as plain Python; notebooks only call it and write the answers. This lets you work in separate notebooks without merge conflicts, and the app imports the exact same feature code, as the brief demands.

```
Emotion_Recognition/
├── config.py              DATA_PATH, SEED=42, TEST_ACTORS={1,16,17,24}, EMOTIONS, C_GRID
├── src/
│   ├── io.py              load_frames(), video_table(), parse_filename()        [A]
│   ├── normalise.py       normalise_d(), normalise_image()                      [A]
│   ├── features.py        measures M1–M6, five statistics, own features         [A base, B own]
│   ├── outliers.py        iqr/zscore flags, inject_outliers(), modified_z()     [B]
│   ├── plotting.py        draw_landmarks(), overlay_video()                     [B]
│   └── evaluation.py      make_pipeline(), run_loso(), nested_cv(), MRMRSelector [A, B adds mRMR]
├── notebooks/
│   ├── 01_data.ipynb               Q1                                           [A + B]
│   ├── 02_quality.ipynb            Q2                                           [A 2.1, B rest]
│   ├── 03_features.ipynb           Q3                                           [A 3.1, B 3.2–3.4]
│   ├── 04_evaluation.ipynb         Q4                                           [A]
│   ├── 05_selection_test.ipynb     Q5                                           [B 5.1–5.5, joint 5.6–5.7]
│   └── 06_deployment.ipynb         Q6–Q7                                        [A 6.1, B 6.3–6.4, joint 6.2, 7]
├── app/                   classifier.py, model.joblib, copy of src modules
├── data/                  gitignored: CSVs + frames.parquet
└── requirements.txt
```

**Contracts to agree on day one** (so neither person blocks the other):

- **Frame table** (`frames.parquet`): columns `video_id, actor, sex, emotion, intensity, statement, repetition, is_test, frame, timestamp, confidence, x_0…x_67, y_0…y_67`. Emotion stored as the code 1–8 plus a label map in `config.py`.
- **`normalise_d(df) -> df`**: same columns, coordinates replaced. Works on one video or many, and on the app's un-normalised MediaPipe frames.
- **`video_features(frames_of_one_video) -> dict`**: returns `{feature_name: value}`. Base features named like `M1_mean`, `M1_std`, `M1_range`, `M1_vel`, `M1_energy`; own features prefixed `own_`. B adds own features by adding functions that this one calls.
- **`build_feature_table(frames) -> DataFrame`**: one row per video, plus `actor` and `emotion` columns for grouping and labels.
- **Report format:** notebooks as the report (answers in Markdown cells under each question number). Less duplication than a separate PDF, and every number is guaranteed to come from the code. Export to HTML at the end.

The brief's README says `classifier.py` must import your own modules from the `app/` folder, so plan to copy (or symlink at build time) `src/` into `app/` before delivery.

## Who does what

Miguel owns 1.1–1.2, 2.1, 3.1, all of Section 4, 6.1; the partner owns 1.3–1.6, 2.2–2.8, 3.2–3.4, 5.1–5.5, 6.3–6.4; the decisions in 5.6, 5.7, 6.2 and 7 are made together. Rough effort is balanced.

| Q | Owner | Needs first | What makes or breaks it |
| --- | --- | --- | --- |
| 1.1 loader + Parquet | A | data download | Read only needed columns; speech files only (field 2 = 01) → 1440 videos; `is_test` flag |
| 1.2 video table | A | 1.1 | Dev actors only; counts per emotion/intensity; frames-per-emotion distribution |
| 1.3 unit of analysis | B | 1.1, 1.2 | Row = frame vs video; **actor** is the dependence group (same face, same habits) |
| 1.4 duration as feature | B | 1.2 | Related? Probably. Legitimate? No — clip length depends on recording protocol, not the face |
| 1.5 landmark plots | B | 1.1, 2.1 | Draw regions as in Fig. 2; neutral vs strong emotion, same actor and statement |
| 1.6 confidence | B | 1.1 | Count low-confidence frames, which videos; say what 2.x does with them |
| 2.1 normalisations | A | 1.1 | Coefficient of variation of 20 actor medians of mouth width; 1280/720 distorts aspect ratio |
| 2.2–2.4 real outliers | B | 2.1 | IQR vs z (k=3, 4) per emotion; trace flagged frames to actors; k-means on (x\_54, y\_54) |
| 2.5–2.8 injected outliers | B | 2.1 | Keep true values and positions; regression residual + modified z; 3 replacements by MAE |
| 3.1 30 base features | A | 2.1 | Velocity and energy on consecutive-frame diffs; check no NaNs; test features computed, not looked at |
| 3.2 ≥ 5 own features | B | 3.1 interface | One sentence each: expression targeted + landmarks used; no file-name fields |
| 3.3 per-emotion plots | B | 3.2 | At least 4 features; which emotions separate, which confuse |
| 3.4 correlation | B | 3.2 | Groups with abs(r) > 0.9; why univariate selection picks redundant features |
| 4.1–4.2 random vs LOSO | A | 3.2 merged | StratifiedShuffleSplit × 30 vs LeaveOneGroupOut; macro F1 per actor |
| 4.3–4.4 compare + confusions | A | 4.2 | Random split leaks the actor → optimistic; row-normalised LOSO confusion matrix |
| 4.5–4.8 choosing C | A | 4.2 | Nested: GridSearchCV + GroupKFold(5) inside each LOSO fold; 4.7 shows the leak with StratifiedKFold |
| 5.1–5.2 PCA | B | 3.2; C from 4.6 for 5.2 | 5.1 exploratory only; 5.2 PCA inside the pipeline at 75 % / 95 % |
| 5.3–5.5 ANOVA vs mRMR | B | C from 4.6 | mRMR wrapped as a transformer fitted per fold; selection counts grouped by M1–M6 |
| 5.6 summary + choice | Both | 4.8, 5.2–5.5 | One table; choose on score **and** simplicity and stability |
| 5.7–5.8 the test | Both | 5.6 frozen | Run once, together; no changes after |
| 6.1 classifier.py | A | 5.7 | Refit on all 1440 videos; `probability=True`; imports `src/` |
| 6.2 recordings | Both | 6.1 | Each records one clip per emotion (16 clips) saying a dataset statement |
| 6.3–6.4 domain shift | B | 6.2 | Feature percentiles vs training; why pixel/image normalisation fails; 15 fps halves T and doubles per-frame velocity |
| 7 conclusions | Both | all | Half a page; what moved the score, what did not; limits of actors-on-request data |

## Timeline

All code is done by Wed 21 Oct and the zip goes in on Fri 23 Oct, leaving Saturday as buffer.

&#91;embedded content: timeline · 9–26 Oct, by owner\]

The partner's track starts on raw CSVs of two dev actors until Miguel's loader and normalisation land on Sun 11; the own features land Thu 15 so Section 4 runs on the final feature set; C is fixed Sun 18 so Section 5 can rerun with it.

## Sync points and joint decisions

Three handoffs keep the tracks unblocked; miss one and the other person waits.

1. **Handoff 1 — Sun 11 Oct:** A merges `frames.parquet` + `normalise_d()`. B can then do all of Q1 exploration and Q2. Until then, B sets up `config.py`, `plotting.py` and the environment from the raw CSVs of a couple of dev actors.
2. **Handoff 2 — Thu 15 Oct:** B merges the own features (3.2) into `features.py`. A needs the final feature set before running 4.1–4.8, otherwise Section 4 must be rerun.
3. **Handoff 3 — Sun 18 Oct:** A publishes the C chosen most often in 4.6 in `config.py` (`C_FIXED`). B reruns 5.2–5.5 with it (code written earlier with a placeholder C = 1).

**Decided together, in a call, not by one person:**

- Which own features to keep (Thu 15) — both must be able to justify each one at the defence.
- The pipeline to deliver in 5.6 (Mon 19), written down with its reason *before* 5.7 is run.
- Running 5.7 (Mon 19): one person shares screen, the other watches, the cell runs once and the result is committed as-is.

**Short daily check-in (10 min):** what I merged, what I'm on, what I'm blocked on. A message in your group chat is enough.

## Git workflow

One branch per question block, merged by pull request that the other person reviews — the review is your defence study, not a formality.

- **Branches:** `a/q1-loader`, `a/q4-eval`, `b/q2-quality`, `b/q3-own-features`, etc. Never commit straight to `main`.
- **Who edits what:** each notebook has one owner at a time (see the structure above). Shared `src/` files: add functions, don't rewrite the other person's without asking.
- **Notebook conflicts:** clear outputs before committing during development (`nbstripout` does it automatically); only the final run on 23 Oct commits outputs.
- **Never commit:** the CSVs, the zip, `frames.parquet`, or anything from the test actors' features. Check `.gitignore` covers `data/`.
- **PR checklist:** runs top to bottom on a fresh kernel; seed from `config.py`; no test actor touched; every figure has its interpreting sentences under the question number.
- **Repo visibility:** it's currently public. Make it private until after the deadline — other groups can see your code, and the dataset licence forbids redistribution anyway.

## Defence prep and delivery checklist

The marks come from reasoning, so the last two days go to explaining, not coding.

**Defence prep (Sat 24 – Sun 25):** each person presents the other's half out loud and gets grilled on it. Questions you should both answer cold:

- Why LOSO and not a random split, and what leaks in 4.7?
- Why is the best point of the C curve (4.5) optimistic, and how does nesting fix it?
- Why does z-score detection fail on injected outliers (2.6) but regression residuals succeed (2.7)?
- Why mRMR differs from ANOVA, linked to the correlation groups of 3.4.
- Why the image-size normalisation would break in the app (6.3), and what 15 fps does to velocity and energy (6.4).
- Every own feature: what it targets and which landmarks.

**Delivery checklist (Fri 23):**

- [ ] Fresh environment (Python 3.10–3.12, `mediapipe==0.10.14`), all notebooks run top to bottom with fixed seeds
- [ ] Data path set in one variable at the top; data not in the zip
- [ ] Every question number answered in Markdown, every figure interpreted in sentences
- [ ] HTML export of each executed notebook
- [ ] `app/` runs with `python app.py`: `classifier.py`, `model.joblib`, imported modules inside
- [ ] `requirements.txt` with pinned versions (`pip freeze` filtered)
- [ ] Zip named `IML_<number1>_<number2>.zip`, uploaded on Inforestudante, download it back and check it opens
