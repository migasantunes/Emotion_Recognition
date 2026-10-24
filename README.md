# Recognising emotion from facial landmark trajectories

Introduction to Machine Learning, 2026/2027 · group 2023210870, 2023211288

The report is the six notebooks in `notebooks/`, executed with their outputs, with an HTML export of each.
The answers are in Markdown cells under the number of each question.

```
config.py        paths (DATA_PATH, PARQUET_PATH), SEED, test actors, emotions, C grid, C_FIXED
src/             the code used by the notebooks: loading, normalisation, features, outliers, plotting, evaluation, selection
notebooks/       01_data (Q1), 02_quality (Q2), 03_features (Q3), 04_evaluation (Q4),
                 05_selection_test (Q5), 06_deployment_7_Answer (Q6 and Q7)
recordings/      the landmarks of our 16 clips recorded with the application (6.2), and two screenshots
app/             the application: classifier.py, model.joblib and the modules it imports (app/src)
requirements.txt the packages used and their versions (Python 3.12)
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

We used AI assistants as a support tool during the project. They helped us plan the work and divide it between the two members. They explained concepts and methods, such as the regression and the modified z-score of 2.7, and helped design some functions of `src/` and some tables and figures. They reviewed the code for errors and suggested improvements. We also used them for brainstorming the own measures M7 and M9, for the LaTeX of the formulas in the table of 3.2, and for a final review of the notebooks against the brief before delivery.