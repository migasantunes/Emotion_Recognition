# Recognising emotion from facial landmark trajectories

**Introduction to Machine Learning** (Introdução à Aprendizagem Computacional)
Master's in Artificial Intelligence · 2026/2027

**Project brief** · Groups of two · published 8 October 2026 · delivery 24 October 2026, 23:59

Marco Simões
Department of Informatics Engineering
Faculty of Sciences and Technology, University of Coimbra

---

## Introduction

### The problem

Facial expressions are one of the main channels through which emotion is communicated. Recognising them automatically has applications in human–computer interaction, in the study of user experience and in health care, among other areas.

This project does not work with images. It works with the geometry of the face: the positions of 68 facial landmarks, tracked in every frame of a video. A video thus becomes a set of trajectories, and the task is to turn those trajectories into a table of features from which a classifier can tell which of eight emotions an actor was portraying.

The objective is not the highest possible score. It is a pipeline in which every decision is justified, evaluated by a protocol that answers the question an application would actually face: how well are the expressions of a person never seen during training recognised?

### The data

The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) [1] contains recordings of 24 professional actors, 12 female and 12 male, vocalising two lexically matched statements in a neutral North American accent. In speech, each actor portrays calm, happy, sad, angry, fearful, surprised and disgusted expressions, each at two levels of intensity, normal and strong, as well as a neutral expression. Every combination is recorded twice.

The RAVDESS Facial Landmark Tracking dataset [2] contains, for each of these videos, the output of the OpenFace 2.1.0 toolkit [3]: one row per frame, at 30 frames per second, for videos of 1280 × 720 pixels. Among other variables, each row holds the image coordinates of the 68 facial landmarks of Figure 2. Only these coordinates are used in this project.

The dataset also contains song recordings. They are excluded. Song has no surprised or disgusted expressions, and one actor recorded no song: with both vocal channels, the channel itself would reveal that two of the eight classes are impossible, a cue that has nothing to do with the face.

### The approach

**Figure 1. The pipeline.** The unit of analysis changes from the frame to the video when the statistics of each measure are computed over the frames of a video.

| Stage | Content | Unit |
| --- | --- | --- |
| CSV file | one per video, 88–157 frames, 136 coordinates | one row = one frame |
| Normalised landmarks | origin at 27, scaled by d (question 2.1) | one row = one frame |
| Six measures | M1–M6 per frame (question 3.1) | one row = one frame |
| Statistics | 5 per measure + own features (questions 3.1–3.2) | one row = one video |
| Feature table | 1200 development + 240 test rows, group = actor | one row = one video |
| Classifier | 8 emotions, SVM, RBF kernel | one prediction per video |

The classifier is fixed: a support vector machine (SVM) with a radial basis function (RBF) kernel, preceded by standardisation. One of its hyperparameters, C, is chosen by the group, and choosing it correctly is part of the work (4). Everything else that decides the result happens before the classifier: how the coordinates are normalised, how anomalies are treated, which features are computed and which of them are kept. The pipeline is designed with 20 of the 24 actors and evaluated once on the other four, set aside from the start (5.7). Finally, it is integrated into a web application that classifies clips recorded with the camera of a computer (6).

No algorithm needs to be implemented from scratch. The work uses Python libraries: NumPy, pandas, SciPy, Matplotlib or seaborn, scikit-learn, and mrmr_selection for 5.4. What is assessed is whether each function is applied correctly, to the right data, for a stated reason.

### Work and delivery

Groups of two. One delivery, on Inforestudante, until 23:59 on Saturday, 24 October 2026. The deliverables are listed in 8 and the rules of assessment in 9.

The questions are numbered. The code is written in Jupyter notebooks or Python files, and the answers are given in one of two forms, chosen by the group:

- **the notebook as the report.** The answers are written in Markdown cells, under the number of each question, next to the code that supports them. Exploration that supports no answer is left out. The notebook is delivered with its outputs.
- **a separate report.** The answers are written in a document, under the number of each question, and the code that produces them is delivered alongside it.

In either form, a figure or a table is not an answer without the sentences that interpret it, and every number and every figure in the report is produced by the code delivered. The suggested outputs at the end of each section indicate what usually supports the answers; they are a guide, not a checklist.

### Calendar

| Date | Event | Place |
| --- | --- | --- |
| 8 October 2026 | Project brief published | UC Student |
| 24 October 2026, 23:59 | Delivery | Inforestudante |
| 26 October 2026 | Individual defence and practical test | UC Figueira Campus |

---

## 1. The dataset

### Obtaining the data

Download the file `FacialTracking_Actors_01-24.zip` (417 MB) from the Zenodo record of the dataset [2], https://doi.org/10.5281/zenodo.3255102. It contains 2452 CSV files, one per video. The dataset is distributed under the Creative Commons BY-NC-SA 4.0 licence: it may be used in this academic work, with attribution, and it is not to be redistributed or included in the delivery.

### File names

Each file is named by seven two-digit fields separated by hyphens (Table 1). Only the files whose vocal channel is 01 (speech) are used: 60 per actor, 1440 in total. Odd-numbered actors are male and even-numbered actors are female [1]. For example, `01-01-06-02-01-02-12.csv` is the audio-visual recording, in speech, of a fearful expression at strong intensity, statement 1, second repetition, by actor 12, who is female.

**Table 1. The fields of the file name [1].**

| Field | Meaning | Codes |
| --- | --- | --- |
| 1 | Modality | 01 full audio-visual (every file of this dataset) |
| 2 | Vocal channel | 01 speech, 02 song (only 01 is used) |
| 3 | Emotion | 01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful, 07 disgust, 08 surprised |
| 4 | Intensity | 01 normal, 02 strong (the neutral expression has normal intensity only) |
| 5 | Statement | 01 "Kids are talking by the door", 02 "Dogs are sitting by the door" |
| 6 | Repetition | 01 first, 02 second |
| 7 | Actor | 01 to 24 |

### Columns

Each file has 712 columns. This project uses `frame`, `timestamp` (in seconds), `confidence` (the confidence of the tracker in the frame) and the 136 coordinates `x_0 … x_67` and `y_0 … y_67`: the position, in pixels, of landmark k in the image, with the origin at the top-left corner and y increasing downwards.

**Figure 2.** The 68 landmarks, numbered as in the columns `x_k` and `y_k`, in one frame of `01-01-08-02-01-01-02.csv` after the normalisation of 2.1. The mouth is enlarged on the right. Origin: landmark 27. Scale: inter-ocular distance d between landmarks 36 and 45.

Landmark regions (standard 68-point layout, as drawn in the figure):

| Region | Landmarks |
| --- | --- |
| Jaw / face contour | 0–16 |
| Right brow (image left) | 17–21 |
| Left brow (image right) | 22–26 |
| Nose bridge | 27–30 |
| Lower nose | 31–35 |
| Right eye (image left) | 36–41 |
| Left eye (image right) | 42–47 |
| Outer lips | 48–59 |
| Inner lips | 60–67 |

### Development and test actors

Four actors form the test set: actors 01 and 17 (male) and 16 and 24 (female), 240 videos. They were drawn at random, two of each sex, one from the easier and one from the harder half of the actors of that sex under a reference pipeline, so that the test set is neither unusually easy nor unusually hard. The other 20 actors, 1200 videos, form the development set.

The test actors are used once, in 5.7. Every other question uses only the development actors: the exploration, the normalisation, the treatment of anomalies, the design and the selection of features, and the choice of C. A decision informed by the test actors, even by looking at their data, makes the test score optimistic, for the same reason as in 4.5.

### Questions

**1.1** Write a function that reads every speech file into a single frame-level table, with one row per frame: the identifier of the video, the fields of the file name that are needed, frame, timestamp, confidence and the 136 coordinates. Report its dimensions, and mark the rows of the four test actors. Reading only the required columns, and saving the table once in a binary format such as Parquet, avoids reading the CSV files again in every session.

**1.2** Build a video-level table of the development actors, with one row per video: actor, sex, emotion, intensity, statement, repetition and number of frames. Report the number of videos per emotion and per intensity, and the distribution of the number of frames per emotion.

**1.3** What does one row of the table of 1.1 represent, and one row of the table of 1.2? Which column identifies groups of rows that are not independent of one another, and why are they not independent?

**1.4** Is the duration of a video related to the emotion portrayed? Would its use as a feature be legitimate in an application that classifies a new recording? Justify.

**1.5** Write a function that draws the 68 landmarks of one frame, joining the points of each facial region as in Figure 2, and a second function that overlays all the frames of one video. For one actor and one statement, compare the neutral video with a video of another emotion at strong intensity.

**1.6** Examine the column confidence. Are there frames in which the tracking appears to have failed? Report what was found and what it implies for 2.

**Suggested outputs**

- the dimensions of both tables and their first rows
- the number of videos per emotion and intensity, and the distribution of the number of frames per emotion
- two overlays from 1.5, side by side

---

## 2. Data quality

Pixel coordinates depend on where the actor sits and on the distance to the camera, neither of which says anything about the expression. The coordinates are therefore normalised first. The anomalies are then examined: first those present in the data and then, in a controlled experiment, outliers injected on purpose, so that their detection and replacement can be measured against a known truth.

### Normalisation

**2.1** Two normalisations of the coordinates are compared. Both translate the origin to landmark 27, which removes the position of the face in the image. The first divides by the inter-ocular distance d, the distance between landmarks 36 and 45 in the same frame:

$$
x'_k = \frac{x_k - x_{27}}{d}, \qquad y'_k = \frac{y_k - y_{27}}{d}, \qquad d = \sqrt{(x_{45} - x_{36})^2 + (y_{45} - y_{36})^2}
$$

The second divides by the dimensions of the image, 1280 × 720 pixels:

$$
x'_k = \frac{x_k - x_{27}}{1280}, \qquad y'_k = \frac{y_k - y_{27}}{720}
$$

Compare the two through the variability between actors in the neutral expression: for each normalisation, compute the median width of the mouth (the distance between landmarks 48 and 54) in the neutral videos of each actor, and the coefficient of variation of these 20 medians. Which normalisation leaves less variation between actors, and why? What does dividing x by 1280 and y by 720 do to the geometry of the face? All the remaining tasks use the normalisation by d.

### Outliers in the data

**2.2** Consider the frames of all development videos together, one row per frame. Draw a box plot per emotion of one normalised coordinate of your choice, for example `x_54`. Then, for every normalised coordinate and every emotion, compute the density of outliers

$$
\rho = 100 \times \frac{n_o}{n}
$$

where n_o is the number of frames flagged and n the number of frames, using (a) the interquartile range (IQR) with Tukey's fences, the rule of the box plot, which flags the values below Q1 − 1.5 IQR or above Q3 + 1.5 IQR, and (b) the z-score with k = 3 and k = 4. For each emotion, report the mean and the standard deviation of ρ over the coordinates. Comment on the differences between the two rules and between emotions.

**2.3** For one coordinate and one emotion, find which actors and which videos the flagged frames come from. Are they errors of the tracker or legitimate, extreme expressions? Should they be removed or corrected before the features are computed? Justify.

**2.4** Apply k-means to the pair (`x_54`, `y_54`) of all development frames, with 2, 4 and 8 clusters, and flag as outliers the frames whose distance to the centroid of their cluster exceeds the mean plus three standard deviations of the distances within that cluster. Compare with 2.2 and illustrate with two-dimensional plots. Optionally, repeat with DBSCAN on a random sample of 20 000 frames; on all frames, its memory requirements can exceed those of a laptop.

### A controlled experiment

When a real frame is flagged, its true value is unknown, so neither the detection nor the replacement can be verified. To study both against a known truth, outliers are injected into one variable, `y_57`, the lower edge of the lower lip, after normalisation.

**2.5** Write a function that receives an array, a rate, k and a random generator, selects at random a fraction of the samples equal to the rate, and replaces each selected value by

$$
p = \mu + s \cdot k \cdot \sigma + q
$$

where μ and σ are the mean and the standard deviation of the array before the injection, s is −1 or +1 with equal probability, and q is drawn uniformly from [0, σ). Inject outliers into 2 % of the frames of `y_57`, with k = 3 and a fixed seed, and keep the original values and the positions of the injected samples.

**2.6** Detect the injected samples with the z-score and k = 3, computed on the contaminated variable. Report the recall and the precision of the detection with respect to the injected positions, and explain the result. Would a smaller k solve the problem, and at what cost?

**2.7** The lower lip moves with its neighbours. Fit a linear regression of `y_57` on `y_56` and `y_58` (scikit-learn) and compute the residual of every frame. Flag the frames whose residual has a modified z-score [4]

$$
M_i = \frac{0.6745 \times (r_i - \mathrm{med}(r))}{\mathrm{MAD}}
$$

greater than 3.5 in absolute value, where r_i is the residual of frame i, med(r) the median of the residuals and MAD their median absolute deviation. Report recall and precision. Why does this detector succeed where that of 2.6 fails? Why is a robust threshold used for the residuals?

**2.8** Replace the values flagged in 2.7 by (a) linear interpolation in time within the same video, (b) the prediction of the regression of 2.7, refitted without the flagged frames, and (c) the mean of the variable. Compare each with the original values through the mean absolute error at the injected positions. Which is best, and why? Which would you choose if the neighbouring landmarks had failed in the same frames?

> **Note.** The controlled experiment studies a method. Sections 3 to 5 use the original coordinates, not the contaminated copy of 2.5.

**Suggested outputs**

- the comparison of the two normalisations
- box plots per emotion, and the outlier densities per emotion for each rule
- a plot of the frames flagged by k-means
- a table with recall and precision of the detectors of 2.6 and 2.7, and the errors of the three replacements of 2.8

---

## 3. Feature construction

From here on, one row is one video. Six measures of the face are computed in every frame (Table 2, Figure 3), and each measure is summarised over the frames of the video by five statistics (Table 3), giving 30 features, to which each group adds features of its own (3.2). Distances are Euclidean distances between normalised landmarks, δ(a, b); heights are differences of normalised y coordinates, with y increasing downwards.

**Table 2. The six measures, computed in every frame from the normalised coordinates.**

| | Measure | Definition | Intended to capture |
| --- | --- | --- | --- |
| M1 | Mouth opening | δ(62, 66) | jaw drop, speech, surprise |
| M2 | Mouth width | δ(48, 54) | smile, stretched lips |
| M3 | Corner lift | mean y′ of 51 and 57 minus mean y′ of 48 and 54; positive when the corners of the mouth are above the centre of the lips | smile, downturned mouth |
| M4 | Brow height | mean y′ of 36–41 minus mean y′ of 17–21, averaged with mean y′ of 42–47 minus mean y′ of 22–26 (eye centroid minus brow centroid, both sides) | raised or lowered brows |
| M5 | Eye opening | ¼ [δ(37, 41) + δ(38, 40) + δ(43, 47) + δ(44, 46)] | widened or narrowed eyes |
| M6 | Brow gap | δ(21, 22) | frowning |

**Figure 3.** The six measures drawn on one frame of `01-01-03-02-01-01-02.csv`. The mouth is enlarged on the right.

**Table 3. The five statistics of a measure m over the T frames of a video.**

| Statistic | Definition |
| --- | --- |
| mean | the mean of m over the T frames |
| standard deviation | the standard deviation of m over the T frames |
| range | the maximum of m minus its minimum |
| mean absolute velocity, v | the mean of \|m(t + 1) − m(t)\| over the T − 1 pairs of consecutive frames |
| movement energy, E | the mean of (m(t + 1) − m(t))² over the T − 1 pairs of consecutive frames |

The velocity and the energy are expressed per frame. Both are computed on the differences between consecutive frames: they describe how much the face moves, not where it is.

**Figure 4.** Measure M1 in three videos of actor 2, statement 1 (neutral `01-01-01-01-01-01-02.csv`, happy strong `01-01-03-02-01-01-02.csv`, surprised strong `01-01-08-02-01-01-02.csv`), plotted over time (about 4 s, M1 in units of d, peaking near 0.31 for surprised). For the surprised video, the dashed line marks the mean and the arrow the range.

### Questions

**3.1** Compute the six measures in every frame and the five statistics for every video, obtaining a table of 30 features with one row per video. Check that it has no missing values. The same functions compute the features of the test videos, which are set aside without being inspected.

**3.2** Propose and compute at least five features of your own, each with a sentence of justification stating the expression it targets and the landmarks it uses. Only information from the landmark files may be used: the fields of the file name other than the emotion (actor, intensity, statement, repetition) are not features.

**3.3** For at least four features, plot the distribution per emotion. Which emotions does each feature separate, and which does it confuse?

**3.4** Compute the correlation matrix of all the features, those of Tables 2 and 3 and those of 3.2, and identify the groups of strongly correlated features, for example with |r| > 0.9. What do these groups imply for a method that selects features one at a time by their individual relevance?

**Suggested outputs**

- the first rows of the feature table
- the plots of 3.3
- the correlation matrix as a heat map, with the groups of 3.4 identified

---

## 4. Evaluation protocol

The classifier is a scikit-learn `Pipeline` of `StandardScaler` and `SVC` with the RBF kernel and default parameters, unless stated otherwise. Every fitted step, including the scaler, belongs to the pipeline, so that it is fitted on the training data of each split only. The main metric is the macro-averaged F1-score; accuracy, the recall of each class and the confusion matrix are reported alongside. The features are all those of Section 3: the 30 of Tables 2 and 3 and those of 3.2. Every protocol of this section uses the development actors only.

**Figure 5. The protocols of the project.** The four test actors (hatched) are set aside from the start: the random split (a, 4.1), leave-one-subject-out (b, 4.2) and the nested procedure used to choose C (c, 4.6) use the 20 development actors; the chosen pipeline is evaluated once on the test actors (d, 5.7).

- **(a) Random stratified 70/30 split of the development actors:** every actor contributes videos to both sets.
- **(b) Leave-one-subject-out (LOSO):** 20 folds, each tests all 60 videos of one unseen development actor.
- **(c) Nested:** inside each outer LOSO fold, C is chosen by GroupKFold(5) over the remaining 19 development actors.
- **(d) Test (question 5.7):** the chosen pipeline is fitted on the 20 development actors and evaluated once on the 4 test actors (1, 16, 17, 24).

### Two protocols

**4.1** Random protocol: a stratified random split with 70 % of the videos for training and 30 % for testing, repeated 30 times with a fixed seed (`StratifiedShuffleSplit`). Report the mean and standard deviation of accuracy and macro F1, and the confusion matrix summed over the repetitions.

**4.2** Leave-one-subject-out (LOSO) protocol: `LeaveOneGroupOut` with the actor as the group, 20 folds, one per development actor. Report the same metrics, and the distribution of the macro F1 over the 20 actors.

**4.3** Compare 4.1 and 4.2. Which question does each protocol answer? Which of them estimates the performance on a person whose face the model has never seen? Explain the difference in terms of 1.3. Which protocol must be used in the rest of the project?

**4.4** Under LOSO, which emotions are confused most often? Is the pattern plausible given the measures of Table 2? Which emotion is the hardest to recognise, and why might that be?

### Choosing a hyperparameter

The RBF SVM has two main hyperparameters, C and γ. This project tunes C over the grid {0.01, 0.1, 1, 10, 100, 1000}; γ keeps its default value (`'scale'`). Tuning γ as well is optional.

**4.5** Report the LOSO macro F1 for each value of C. Why is the best value of this curve an optimistic estimate of the performance of the pipeline?

**4.6** Nested procedure (Figure 5c). In each outer LOSO fold, choose C with `GridSearchCV` on the training actors only, with an inner `GroupKFold(5)` that keeps the videos of each actor in the same inner fold; then evaluate the chosen model on the held-out actor. Report the outer macro F1 (mean and standard deviation), the value of C chosen in each outer fold and the mean of the best inner scores [5].

**4.7** Repeat 4.6 with an inner `StratifiedKFold(5)` that ignores the actor. Compare with 4.6: the values of C chosen, the inner estimates and the outer scores. Explain what happened. What does it imply for every other decision made with cross-validation in this project?

**4.8** Repeat 4.6 with the 30 features of Tables 2 and 3 only. Did the features of 3.2 improve the result? Is the difference larger than the variation between folds?

**Suggested outputs**

- a table with both protocols: accuracy and macro F1 (mean and standard deviation)
- the LOSO confusion matrix, normalised by row, and the distribution of the score over actors
- the curve of 4.5
- a table comparing 4.6 and 4.7: values of C chosen, inner estimate, outer score

---

## 5. Reduction, selection and test

Every transformation and every selection is fitted on the training actors of each LOSO fold, and never on all videos. In this section C is fixed to the value chosen most often in 4.6; state in the report what this simplification may cost. The features are all those of Section 3.

### Principal component analysis

**5.1** Apply principal component analysis (PCA) to the standardised features of all development videos, for exploration only. Plot the cumulative explained variance and report the number of components that explain 75 % and 95 % of the variance. Which measures dominate the first two components? Plot the videos on the first two components, coloured by emotion.

**5.2** Under LOSO, evaluate the pipelines `StandardScaler → PCA → SVC` keeping 75 % and 95 % of the variance. Compare with all the features. Does compressing the representation help the classifier? Why?

### Feature selection

**5.3** Univariate selection: `SelectKBest` with the ANOVA F-statistic (`f_classif`) inside the pipeline, with K = 5, 10 and 15. Report the LOSO macro F1.

**5.4** Minimum redundancy, maximum relevance (mRMR) [6]: select K = 5, 10 and 15 features with `mrmr_classif` from the `mrmr_selection` package (`pip install mrmr_selection`), on the training actors of each LOSO fold. With its default arguments, the package measures the relevance of a feature by its F-statistic and its redundancy by its mean absolute Pearson correlation with the features already selected, and selects at each step the feature that maximises the quotient of the two, a variant proposed for continuous features in [7]. The function is not a scikit-learn transformer: the selection is performed explicitly within each fold, or wrapped in a transformer. Compare with 5.3.

**5.5** For K = 10, count how many times each feature is selected over the 20 folds, by the ANOVA selection and by mRMR, and group the counts by measure (M1 to M6, and the features of 3.2). Relate the differences to 3.4.

**5.6** Summarise in one table every configuration evaluated under LOSO: all the features, the 30 features of Tables 2 and 3 (4.8), PCA, ANOVA and mRMR with each K, with the number of features and the macro F1 (mean and standard deviation). Which pipeline would you deliver, and why? Is the score the only criterion?

### The test

**5.7** Fit the pipeline chosen in 5.6, with the value of C used in this section, on the 1200 development videos, and evaluate it once on the 240 videos of the test actors. Report accuracy and macro F1, the confusion matrix and the macro F1 of each test actor. From this point on, no decision of Sections 1 to 5 is changed: a pipeline revised after seeing its test score no longer has an unbiased test score.

**5.8** Compare the test score with the LOSO estimate of the same pipeline in 5.6. Which of the two is an unbiased estimate of the performance of the pipeline delivered, and why? From the distribution of the macro F1 over actors in 4.2, estimate how much the mean over four actors would vary from one choice of test actors to another. Is the difference between the two scores larger than that variation?

**Suggested outputs**

- the explained-variance plot and the projection on the first two components
- the selection frequencies of 5.5, grouped by measure
- the summary table of 5.6
- the test results of 5.7: accuracy, macro F1, the confusion matrix and the score of each test actor

---

## 6. Deployment

A model is useful only inside an application, and the conditions there are not those of the data. A web application is provided on the course page, in the folder `app`: it records a short clip with the camera of the computer, or accepts a video file, extracts the landmarks of every frame and shows the emotion predicted by a function, `predict`, that each group implements (Figure 6).

The landmarks are extracted with MediaPipe [8], whose face mesh is mapped to the 68 points of Figure 2, converted to pixels and resampled to 30 frames per second. The clip thus reaches `predict` as a table in the format of the CSV files of the dataset, with coordinates that are not normalised. Only the file `classifier.py` is changed; the file `README.md` of the folder describes how to install and run the application.

**Figure 6. The application.** Each group writes `classifier.py`; the rest is provided.

| Step | Content | File | Written by |
| --- | --- | --- | --- |
| Camera | clip of 3–5 s recorded in the web page | `app.py` | provided |
| Landmarks | MediaPipe, one face, 68 points (Figure 2), pixels | `landmarks.py` | provided |
| Frame table | resampled to 30 fps, format of the dataset, not normalised | `landmarks.py` | provided |
| Features | normalisation and features, same code as in training | `classifier.py` | the group: `predict(model, frames)` |
| Model | the pipeline of 5.6, fitted on the 1440 videos | `classifier.py` | the group: `predict(model, frames)` |
| Web page | probability of each emotion, landmarks, CSV | `app.py` | provided |

**6.1** Fit the pipeline chosen in 5.6 on all 1440 videos, the development and the test actors together: once 5.7 has been answered, the test actors have served their purpose, and a model fitted on more actors is expected to generalise at least as well. Save it (for example with `joblib`), and implement `load_model` and `predict` in `classifier.py`. `predict` normalises the coordinates and computes the features with the same functions used for training, imported from the group's own modules, and returns the probability of each emotion. An `SVC` returns probabilities only if it is created with `probability=True`.

**6.2** With the application, record at least one clip per emotion by each member of the group, saying one of the two statements of the dataset. Report the predictions in a table, with a screenshot of the page for one of them. Which emotions are recognised, and which are not?

**6.3** The application differs from the data in the tracker (MediaPipe instead of OpenFace), in the camera and the recording conditions, and in the person. For the clips of 6.2, locate the value of each feature in its distribution in the training data, for example as a percentile. Which features fall outside the range of the training data, and why? Which of the two normalisations of 2.1 would have failed in the application, and why?

**6.4** The application resamples the landmarks to 30 frames per second. Which statistics of Table 3 would change, and in which direction, if the clips of a camera recording at 15 frames per second reached `predict` without resampling? Why must the frame rate seen by the model match that of the training data?

**Suggested outputs**

- `classifier.py` and the saved model
- the table of predictions of 6.2 and one screenshot
- the comparison of 6.3

---

## 7. Conclusions

**7.1** In at most half a page, state which decisions of the pipeline changed the result most, and which did not change it.

**7.2** The data were recorded by 24 professional actors, portraying emotions on request while reading two statements, facing one camera. Which of these conditions limit what the LOSO and test scores say about the users of a real application?

---

## 8. Deliverables

One file, `IML_<number1>_<number2>.zip`, submitted on Inforestudante until 23:59 on Saturday, 24 October 2026, containing:

- **the report:** either the notebook that answers the questions, executed, with its outputs, together with an HTML or PDF export of it, or a separate report in PDF;
- **the code:** the notebooks or Python files that produce every result, including any modules they import;
- **the folder `app`,** with `classifier.py` implemented, the saved model and the modules it imports, so that the application runs with `python app.py`;
- **a `requirements.txt` file** listing the packages used and their versions.

The data are not included. The path to the data is set in one variable at the beginning of the code. Every number and every figure in the report is produced by the code delivered, which runs from start to end in a clean environment, with fixed random seeds. A complete solution runs in a few minutes on a laptop once the frame table of 1.1 has been saved. The application may be demonstrated at the defence.

---

## 9. Assessment

**The practical component.** The project and the practical test form the practical component of the course: 40 % of the final grade, 8 out of 20 values. The project is worth 5 values and the practical test 3. The minimum is 35 % in each component of the assessment, with no compensation between components.

**The defence.** Individual and compulsory, on the group's own project. Its result, a percentage, multiplies the mark of the project. A defence of 0 % gives a project mark of 0.

**The practical test.** Individual, on the day of the defence, with mostly multiple-choice and short-answer questions.

**The mark of the project.**

| Component | Weight |
| --- | --- |
| Data, unit of analysis and data quality (1, 2) | [__] |
| Feature construction (3) | [__] |
| Evaluation protocol and choice of C (4) | [__] |
| Reduction, selection and test (5) | [__] |
| Deployment (6) | [__] |
| Conclusions (7) | [__] |
| Report and code (8) | [__] |

Marks are earned by the reasoning, not by the score. A decision that lowered the score and was kept for a stated reason is worth more than one that raised it and cannot be explained. Two groups with the same score do not necessarily receive the same mark.

Any result in which a fitted step (scaler, transformation, selection, hyperparameter choice or classifier) used the videos of the actor being tested voids the question it belongs to.

Any use of the test actors before 5.7, in the exploration, the normalisation, the treatment of anomalies, the design or the selection of features or the choice of C, voids the questions it affects.

---

## References

[1] S. R. Livingstone, F. A. Russo, "The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English", *PLOS ONE*, vol. 13, no. 5, e0196391, 2018. https://doi.org/10.1371/journal.pone.0196391

[2] R. Swanson, S. R. Livingstone, F. A. Russo, "RAVDESS Facial Landmark Tracking", version 1.0, Zenodo, 2019. https://doi.org/10.5281/zenodo.3255102

[3] T. Baltrusaitis, A. Zadeh, Y. C. Lim, L.-P. Morency, "OpenFace 2.0: Facial Behavior Analysis Toolkit", in *13th IEEE International Conference on Automatic Face & Gesture Recognition (FG 2018)*, pp. 59–66, 2018. https://doi.org/10.1109/FG.2018.00019

[4] B. Iglewicz, D. C. Hoaglin, *How to Detect and Handle Outliers*, The ASQC Basic References in Quality Control: Statistical Techniques, vol. 16, E. F. Mykytka (Ed.), 1993.

[5] S. Varma, R. Simon, "Bias in error estimation when using cross-validation for model selection", *BMC Bioinformatics*, vol. 7, art. 91, 2006. https://doi.org/10.1186/1471-2105-7-91

[6] H. Peng, F. Long, C. Ding, "Feature selection based on mutual information criteria of max-dependency, max-relevance, and min-redundancy", *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 27, no. 8, pp. 1226–1238, 2005. https://doi.org/10.1109/TPAMI.2005.159

[7] C. Ding, H. Peng, "Minimum redundancy feature selection from microarray gene expression data", *Journal of Bioinformatics and Computational Biology*, vol. 3, no. 2, pp. 185–205, 2005. https://doi.org/10.1142/S0219720005001004

[8] Y. Kartynnik, A. Ablavatski, I. Grishchenko, M. Grundmann, "Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs", arXiv:1907.06724, 2019. https://arxiv.org/abs/1907.06724
