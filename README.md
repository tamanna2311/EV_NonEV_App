# EV vs Non-EV Detector


Live App: [Open EV Non-EV Detector](https://ev-nonev-app.onrender.com)

This project detects whether a vehicle is EV-like or Non-EV-like using accelerometer data.

Streamlit app for classifying accelerometer recordings as EV-like or Non-EV-like.


 ## Render Deployment

Use these settings when creating the app:

- Repository: `tamanna2311/EV_NonEV_App`
- Branch: `main`
- Main file path: `app.py`
- Python version: `3.12`

The app needs `best_ev_nonev_model.pkl` and `feature_columns.pkl` in the repository root.

## Input files

Upload a `.csv` file with accelerometer columns `x`, `y`, and `z`. A time column can be named `time_sec`, `millisecond`, `milliseconds`, or `time`.

The app also accepts RAW_ACCELEROMETERS-style `.txt` files with time in column 0 and accelerometer axes in columns 8, 9, and 10.
