# IML project: the application

A web page that records a short clip with the computer's camera, extracts the 68
facial landmarks of every frame and shows the emotion predicted by **your** pipeline.

```
app/
├── app.py                        the web application 
├── landmarks.py                  landmark extraction with MediaPipe, 68 points, 30 fps
├── models/
│   └── face_landmarker.task      the MediaPipe Face Landmarker model, used with mediapipe newer than 0.10.14
├── classifier.py                 our code: load_model() and predict(model, frames)
├── model.joblib                  the pipeline, fitted on all 1440 videos (written by notebooks/06_deployment_7_Answer.ipynb)
│                                 
├── src/                          our own modules, the same code used for training
│   ├── normalise.py              normalise_d: origin at landmark 27, scaled by the inter-ocular distance d
│   └── features.py               video_features: the measures and their statistics (55 features)
├── requirements.txt              the packages needed
└── README.md                     this file
```


## Install (In case didn't run the notebooks) and run

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open the address printed in the terminal (by default http://127.0.0.1:7860), allow the browser to use the camera, record a clip of 3 to 5 seconds facing the camera, and press **Classify**.\
A video file can be uploaded instead of recording one, which is also the way to test the application on a computer without a camera.

Everything runs on your computer: the clips are not sent anywhere.

## What the application does to the clip

1. Reads every frame of the clip.
2. Finds one face per frame with MediaPipe and keeps 68 of its points, mapped to the numbering of Figure 2 of the brief (`MP68` in `landmarks.py`). Frames without a face are skipped.
3. Converts the points to pixels, with the origin at the top-left corner and y
   increasing downwards, as in the dataset. They are **not** normalised.
4. Resamples every coordinate to 30 frames per second by linear interpolation in
   time, as in the dataset. `confidence` is set to 1.
5. Calls `predict` and shows the probabilities, the landmarks of the middle frame,
   and the table of landmarks as a CSV file that can be downloaded.

## MediaPipe versions

* With `mediapipe` 0.10.14 or older, the Face Mesh model installed with the package
  is used.
* With newer versions, the Face Landmarker model is downloaded once, at the first
  run, to `models/face_landmarker.task`. If the download fails, get the file from
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
  and put it in `models/`.

Tested with Python 3.10, mediapipe 0.10.14, gradio 6.29 and OpenCV 4.11. If the
installation fails with a recent version of Python, create the environment with
Python 3.10 to 3.12 and install `mediapipe==0.10.14`.