# Work Tracker + AI use
This file will be used to track the work done and also the use of AI
## Tracker
### Config.py
- Added every constants that will be used in the project, lacks the parquet path.
- pip install -e .

### Q2.2 (B)
- `src/outliers.py`: `iqr_flags(X, k=1.5)` and `zscore_flags(X, k)`. They receive the frames of one emotion x the 136 coordinates and return a True/False matrix of the same shape (True = flagged). Statistics computed per column (`axis=0`), `ddof=0` everywhere.
- `x_27`/`y_27` are always 0 after normalisation (std = 0): `np.divide(..., out=zeros, where=sigma > 0)` leaves z = 0 there instead of 0/0 = NaN.
- Notebook 02: loop over 3 rules x 8 emotions -> `density["IQR"]`, `density["z (k=3)"]`, `density["z (k=4)"]`, each 8 emotions x 136 coordinates + `mean` and `std` columns.
- **To review later:** the density loop line by line (`rules.items()`, the lambdas, `.loc` per emotion, `.mean(axis=0)` = fraction of True, `.T`, `axis=1` for mean/std).
- **Still to do:** comment on the differences between rules and between emotions, plus a summary table with only mean/std of the 3 rules.

## Teacher's suggestions
## Nota 2.4.
 - No dbscan usar apenas uma amostra pequena 1000 ou 2000 por causa do custo computacional

## AI Use
- Helped create the plan used to complete the project, in order to have a clear view and organized work devition between the team members.
- Q2.2: helped design and explain `outliers.py`.

># IMPORTANT: 
>### The app must run standalone with python app.py and import its modules from its own folder. For the same reason, keep normalise.py and features.py free of import config, or they break when copied into app/.
>### For delivery, write requirements.txt with the same package names pinned as name==version, taking the versions from pip freeze. Don't paste the whole freeze: it has about 100 transitive packages and a -e git+https://github.com/... line pointing at your repo.