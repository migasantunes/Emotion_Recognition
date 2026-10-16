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
- Summary table (`pd.concat` of the mean/std columns, 8 x 6) and answer done. Theoretical rates for normal data (IQR 0.70 %, z3 0.27 %, z4 0.006 %) are cited from [4], not computed.

### Q2.3 (B)
- Choice: `y_57` in sad with the IQR (variable of 2.5-2.8; sad has outliers on both sides of the box; IQR = the points of the box plot).
- `sad` gets two new columns: `flag` (IQR, fences within sad) and `side` (low/high, split by the median = middle line of the box plot). `flagged = sad[sad["flag"]]` keeps the rows, all columns.
- Tables per actor (`crosstab` + `reindex` so the 20 actors appear) and per video (`crosstab` + `groupby("video_id")` aligned by index).
- Findings: 220 flagged frames, concentrated in actors 21 and 20; no video mixes low and high. Actor 21 = wide-open mouth (legit); actors 21, 20, 11 are extremes in neutral too (face shape); actor 15 = head bent down at the end of the video, confidence unreliable.
- Decision: keep all flagged frames (option 1); actor 15 is a limitation.
- `pd.set_option("display.show_dimensions", True)` in the imports, so every table shows its shape.
- **To review later:** `crosstab`, `reindex(fill_value=0)`, alignment by index when assigning a Series to a column, `set_index("frame")` + `.loc[frame]`.

### Q2.4 (B)
- `src/outliers.py`: `kmeans_flags(X, k, seed)` (distance to own centroid > mean + 3 std within the cluster, `n_init=10`) and `dbscan_flags(X, eps, min_samples=4)` (noise = label -1). Both return `flags, labels`.
- k-means on (`x_54`, `y_54`) of all development frames, 8 emotions together, no scaling (same unit, d). Compared with the IQR recomputed on the same pooled frames (outside in x_54 or y_54).
- DBSCAN on `df_d.sample(n=2000, random_state=SEED)`; `eps` from the k-distance plot (distance to the 4th point counting itself), tested 0.015, 0.02, 0.03. `pos = df_d.index.get_indexer(sample.index)` aligns the sample with the flags of all frames; `_s` = sample.
- Findings: the cloud is diagonal, so the IQR rectangle flags its dense ends; k = 2 is almost a subset of the IQR; larger k flags normal frames at the edges of small central clusters; all methods flag the trails (consecutive frames of single videos?); DBSCAN noise depends strongly on eps; the trails are not in the sample.
- **To review later:** k-means iterations inside `.fit` (k-means++, n_init), `cluster_centers_[labels]`, `flags[in_c] = ...`, `.labels_` and the trailing `_`, `np.bincount(...).argmax()`, `get_indexer`.

## Teacher's suggestions
## Nota 2.4.
 - No dbscan usar apenas uma amostra pequena 1000 ou 2000 por causa do custo computacional

## AI Use
- Helped create the plan used to complete the project, in order to have a clear view and organized work devition between the team members.
- Q2.2: helped design and explain `outliers.py`.
- Q2.3: helped design and explain the tables and figures.
- Q2.4: helped design and explain `kmeans_flags`, `dbscan_flags` and the comparison tables and figures.

># IMPORTANT: 
>### The app must run standalone with python app.py and import its modules from its own folder. For the same reason, keep normalise.py and features.py free of import config, or they break when copied into app/.
>### For delivery, write requirements.txt with the same package names pinned as name==version, taking the versions from pip freeze. Don't paste the whole freeze: it has about 100 transitive packages and a -e git+https://github.com/... line pointing at your repo.