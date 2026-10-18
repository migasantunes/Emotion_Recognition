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

### Q2.5 (B)
- `src/outliers.py`: `inject_outliers(x, rate, k, rng)` returns a contaminated copy and the positions replaced; `x` is not changed. `round(rate * n)` positions without repetition, p = mu + s*k*sigma + q with mu, sigma of the clean array (`ddof=0`).
- `y_57` of all development frames pooled, rate 0.02, k = 3, `rng = np.random.default_rng(SEED)` -> 2644 injected frames (1335 with s = -1, 1309 with s = +1). `y_57_c` = contaminated array; `df_c` = copy of `df_d` with `y_57` replaced and an `injected` column; `inj_pos` = positions (hence `iloc`). `df_d` keeps the original values.
- q is added on both sides, so s = -1 lands between mu - 3 sigma and mu - 2 sigma (inside the 3 sigma band) and s = +1 between mu + 3 sigma and mu + 4 sigma.
- Histogram lines (2.5) = where the injection placed the values (clean mu, sigma); not the detector thresholds.

### Q2.6 (B)
- `src/outliers.py`: `detection_scores(flags, truth)` -> flagged, TP, FP, FN, recall, precision (plain numpy, no sklearn; precision = nan if nothing flagged).
- z-score on the contaminated `y_57` for k = 3, 2.5, 2, 1.5, TP split by side (`below` = injected value below the clean mean). Table of mean/std clean vs contaminated; histogram of z with lines at +-2, +-3.
- Findings: k = 3 -> recall 0.368, precision 0.610; no s = -1 caught; injected values inflate the std (0.0748 -> 0.0808), so part of the s = +1 falls below z = 3 (masking). Smaller k trades missed outliers for false alarms.
- Histogram lines (2.6) = detector thresholds, on the scale of the contaminated mean/std ("same values, stretched ruler").
- **To review later:** `pd.DataFrame.from_dict(rows, orient="index")`, `&` / `~` on boolean arrays, `Series` vs `DataFrame` (`.to_frame`).

### Q2.7 (B)
- `src/outliers.py`: `median_mad(r)` -> median and MAD; `modified_z(r)` = 0.6745 (r - med) / MAD, calls `median_mad`.
- `LinearRegression` of the contaminated `y_57` on `y_56` and `y_58`, all frames: `fit` learns a, b, c (least squares, uses `y_57`), `predict` applies them row by row, residual = real - predicted. R² written by hand: 1 - sum(resid²) / sum((y - mean)²). a = 0.0071, b = 0.481, c = 0.522, R² = 0.83.
- Spread table: centre, spread (std vs MAD / 0.6745) and cutoff = 3.5 x spread (0.117 vs 0.0106). 0.6745: on normal data MAD ~ 0.6745 std, so MAD / 0.6745 is on the scale of the std.
- Detection table: z of y_57 (2.6), plain z of residuals (3.5), modified z of residuals (3.5). Scatter contaminated vs predicted `y_57`, band = cutoff of the table.
- Findings: modified z -> recall 0.998, precision 0.964 (6 FN, 98 FP); plain z of the residuals misses 294 (masking again). Not in the notebook: 21 of the 623 FP of 2.6 are also flagged in 2.7, so the answer says "usually small".
- **To review later:** least squares, b = cov / var with one predictor (normal equations with two, not needed), `.loc[row, column]`.

### Q2.8 (B)
- `src/outliers.py`: `interpolate_flags(df, col, flags)`: flagged -> NaN, linear interpolation within each video (`groupby("video_id").transform`), nearest value copied at the start/end (`limit_direction="both"`). Frames are consecutive and sorted within each video.
- Imputation: the 2736 frames flagged in 2.7 are treated as NaN (not the injected ones). (a) interpolation; (b) regression refitted on the frames not flagged, predicts the flagged; (c) mean of the frames not flagged. MAE at the 2644 injected positions vs `df_d["y_57"]`; the 6 missed keep their injected value.
- Findings: (b) 0.0021, (a) 0.0045, (c) 0.0580. Neighbouring landmarks (56, 58, same frame) = (b); neighbouring frames (same landmark 57) = (a). If the neighbouring landmarks failed -> (a). The cause of (a) > (b) is argued, not measured.
- Summary table (`final`) = the brief's suggested output: recall/precision of 2.6 and 2.7 + MAE of 2.8.

## Teacher's suggestions
### Nota 2.4.
 - In DBScan use only a small pool of samples around 1000 or 2000 because of computacional cost

# Own Features
>< = smaller, less \
>\> = bigger, more \
> y1-y5 = y1 to y5

M7: Mouth corner to eye corner | Happy | (euc_dist(48,36) + euc_dist(54,45))/2, <value = >Happy\
M8: Lower Lip closer to Lower Nose | Sad | mean(y55-y59) - mean(y31-y35), <value = >Sad\
M9: Brow slope | Angry | mean(y21,y22) - mean(y17, y26), >value = >Angry\
M10: Higher Lip closer to Lower Nose | Disgust | mean(y49-y53) - mean(y31-y35), <value= >disgusted\
M11: Brow shape | Surprised or fearful | mean hight between brow extremeties minus the hight of themid point of the brow, averaging with both brows, mean(y17, y21) − y19
/>value = >round = >surprised | <value = >flat = >fearful. Points used: 17, 19, 21, 22, 24, 26. \

## AI Use
- Helped create the plan used to complete the project, in order to have a clear view and organized work devition between the team members.
- Used for code quality review, to check for errors and suggest improvements.
- Used for brainstorming for the measures M7 and M9.
- LaTeX formatting for the measure formulas inside the table in the notebook.
- Q2.2: helped design and explain `outliers.py`.
- Q2.3: helped design and explain the tables and figures.
- Q2.4: helped design and explain `kmeans_flags`, `dbscan_flags` and the comparison tables and figures.
- Q2.5: helped design and explain `inject_outliers` and the check table and histogram.
- Q2.6: helped design and explain `detection_scores`, the detection tables and histogram.
- Q2.7: explained the regression and the modified z-score; helped design `median_mad`, `modified_z`, the tables and the scatter plot.
- Q2.8: helped design and explain `interpolate_flags`, the three replacements, the MAE and summary tables and the figure.