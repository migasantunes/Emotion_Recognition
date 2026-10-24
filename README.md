# Recognising emotion from facial landmark trajectories

Introduction to Machine Learning, 2026/2027 · group 2023210870, 2023211288

The report is the six notebooks in `notebooks/`, executed with their outputs, with an HTML export of each.
The answers are in Markdown cells under the number of each question.

```
IML_2023210870_2023211288/
├── README.md                     this file
├── config.py                     DATA_PATH, PARQUET_PATH, SEED = 42, TEST_ACTORS = [1, 16, 17, 24],
│                                 EMOTIONS (code → label), C_GRID, C_FIXED = 1, K_GRID = [5, 10, 15]
├── pyproject.toml                makes config and src importable from the notebooks (pip install -e .)
├── requirements.txt              the packages used and their versions (Python 3.12)
│
├── src/                          the code the notebooks call; the notebooks only run it and answer
│   ├── loading.py                parse_filename, read_file, load_frames (frame table → Parquet), video_table   Q1
│   ├── normalise.py              normalise_d (origin 27, scale d), normalise_image (1280 × 720)                Q2.1
│   ├── outliers.py               iqr_flags, zscore_flags, kmeans_flags, dbscan_flags, inject_outliers,
│   │                             detection_scores, median_mad, modified_z, interpolate_flags                   Q2.2–2.8
│   ├── plotting.py               frame_xy, draw_landmarks, overlay_video                                       Q1.5
│   ├── features.py               measures (M1–M6), own_measures, video_features (5 statistics per measure),
│   │                             build_feature_table (one row per video)                                       Q3
│   ├── evaluation.py             evaluation (LOSO / random split), nested_evaluation (C chosen per fold),
│   │                             MRMRSelector (mRMR as a scikit-learn transformer)                             Q4, Q5
│   └── selection.py              PCA loadings by measure and statistic, PCA components per fold               Q5.1–5.2
│
├── notebooks/                    the report: answers in Markdown under each question, executed, + HTML export
│   ├── 01_data                   Q1  frame and video tables, counts, landmark overlays, confidence
│   ├── 02_quality                Q2  normalisations, outliers (IQR, z-score, k-means), controlled experiment
│   ├── 03_features               Q3  the 30 base features, own features, distributions, correlation
│   ├── 04_evaluation             Q4  random vs LOSO, the C curve, nested CV, GroupKFold vs StratifiedKFold
│   ├── 05_selection_test         Q5  PCA, ANOVA, mRMR, summary table, the test on actors 1, 16, 17, 24
│   └── 06_deployment_7_Answer    Q6, Q7  final model on 1440 videos, recordings, domain shift, conclusions
│                                 (each as .ipynb and .html)
│
├── recordings/                   our 16 clips from the application (6.2): one per emotion per member
│   ├── andre_<emotion>.csv       8 landmark tables, format of the dataset
│   ├── miguel_<emotion>.csv      8 landmark tables, format of the dataset
│   └── *__screenshot.png         one screenshot of the page per member
│
├── app/                          the application, runs with  cd app && python app.py
│   ├── app.py                    web page
│   ├── landmarks.py              MediaPipe → 68 points, 30 fps
│   ├── models/face_landmarker.task   MediaPipe model
│   ├── classifier.py             load_model, predict: our code
│   ├── model.joblib              the chosen pipeline fitted on all 1440 videos (written by notebook 06)
│   ├── src/                      copies of src/normalise.py and src/features.py, imported by classifier.py
│   ├── requirements.txt          the app's packages
│   └── README.md                 how to install and run the app
│
└── data/                         not included; you create it (see below)
    ├── FacialTracking_Actors_01-24/   the CSV files from Zenodo
    └── Parquet/frame_table.parquet    written by the first call to load_frames
```


## Running the notebooks

1. Create an environment with Python 3.12 and install the packages and the project itself, so that the notebooks can import `config` and `src`:

   ```
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .
   ```

2. Download `FacialTracking_Actors_01-24.zip` from https://doi.org/10.5281/zenodo.3255102 and extract it to `data/FacialTracking_Actors_01-24/`. The data are not included. The path is set in `DATA_PATH` in `config.py`.

3. Run the notebooks in order, from 01 to 06. The first call to `load_frames` reads the CSV files and saves the frame table to `data/Parquet/frame_table.parquet`. Later calls read that file. Notebook 06 fits the final model on the 1440 videos and saves it to `app/model.joblib`.

All random seeds are taken from `SEED` in `config.py`.

## Running the application

```
cd app
python app.py
```

See `app/README.md` for the details.

## Use of AI

- We used AI assistants as a support tool during the project. 
- They helped us plan the work and divide it between the two members. 
- They explained concepts and methods and helped design some functions of `src/`, tables and figures. 
- They reviewed the code for errors and suggested improvements. 
- We also used them for brainstorming the own measures M7 and M9
- The LaTeX of the formulas in the table of 3.2
- For a final review of the notebooks against the brief before delivery.