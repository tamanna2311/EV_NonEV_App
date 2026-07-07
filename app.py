
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq


# ============================================================
# SETTINGS
# Must be same as training code
# ============================================================

SAMPLING_RATE = 100
WINDOW_SECONDS = 3
WINDOW_SIZE = SAMPLING_RATE * WINDOW_SECONDS

LOW_FREQ_CUTOFF = 3.0
FFT_MIN_FREQ = 3.0
FFT_MAX_FREQ = 45.0
STOP_PERCENTILE = 50


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("best_ev_nonev_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")


# ============================================================
# DATA CLEANING
# ============================================================

def clean_csv_file(df):
    df.columns = [str(col).lower().strip() for col in df.columns]

    if "time_sec" not in df.columns:
        if "millisecond" in df.columns:
            ms = pd.to_numeric(df["millisecond"], errors="coerce")
            df["time_sec"] = (ms - ms.iloc[0]) / 1000

        elif "milliseconds" in df.columns:
            ms = pd.to_numeric(df["milliseconds"], errors="coerce")
            df["time_sec"] = (ms - ms.iloc[0]) / 1000

        elif "time" in df.columns:
            df["time_sec"] = pd.to_numeric(df["time"], errors="coerce")

        else:
            df["time_sec"] = df.index / SAMPLING_RATE

    required_cols = ["time_sec", "x", "y", "z"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df = df[required_cols].copy()

    df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["z"] = pd.to_numeric(df["z"], errors="coerce")

    df = df.dropna()
    df = df.sort_values("time_sec").reset_index(drop=True)
    df = df.drop_duplicates(subset="time_sec").reset_index(drop=True)

    df["time_sec"] = df["time_sec"] - df["time_sec"].iloc[0]

    return df


def clean_txt_file(uploaded_file):
    raw_df = pd.read_csv(uploaded_file, sep=r"\s+", header=None)

    if raw_df.shape[1] < 11:
        raise ValueError(
            "TXT file does not have enough columns. Expected RAW_ACCELEROMETERS style file."
        )

    TIME_COL = 0
    XYZ_COLS = [8, 9, 10]

    df = raw_df.iloc[:, [TIME_COL] + XYZ_COLS].copy()
    df.columns = ["time_sec", "x", "y", "z"]

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()

    df = df.sort_values("time_sec").reset_index(drop=True)
    df = df.drop_duplicates(subset="time_sec").reset_index(drop=True)

    df["time_sec"] = df["time_sec"] - df["time_sec"].iloc[0]

    return df


def resample_to_100hz(df):
    df = df.copy()
    df = df.sort_values("time_sec").reset_index(drop=True)

    if len(df) < 2:
        raise ValueError("File is too short.")

    start_time = df["time_sec"].iloc[0]
    end_time = df["time_sec"].iloc[-1]

    if end_time <= start_time:
        raise ValueError("Invalid time column.")

    target_gap = 1 / SAMPLING_RATE

    new_time = np.arange(start_time, end_time, target_gap)

    if len(new_time) < WINDOW_SIZE:
        raise ValueError("File is too short. Need at least 3 seconds of data.")

    x_new = np.interp(new_time, df["time_sec"], df["x"])
    y_new = np.interp(new_time, df["time_sec"], df["y"])
    z_new = np.interp(new_time, df["time_sec"], df["z"])

    new_df = pd.DataFrame({
        "time_sec": new_time - new_time[0],
        "x": x_new,
        "y": y_new,
        "z": z_new
    })

    return new_df


# ============================================================
# PREPROCESSING
# ============================================================

def add_magnitude_and_remove_gravity(df):
    df = df.copy()

    df["magnitude"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)

    rolling_mean = df["magnitude"].rolling(
        window=SAMPLING_RATE,
        center=True,
        min_periods=1
    ).mean()

    df["vibration_no_gravity"] = df["magnitude"] - rolling_mean

    return df


def highpass_filter(signal, cutoff_freq, sampling_rate):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist

    b, a = butter(
        N=4,
        Wn=normal_cutoff,
        btype="highpass"
    )

    return filtfilt(b, a, signal)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(df):
    features = []

    df = df.sort_values("time_sec").reset_index(drop=True)
    df = add_magnitude_and_remove_gravity(df)

    window_infos = []

    for start in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE):
        window = df.iloc[start:start + WINDOW_SIZE]

        signal = window["vibration_no_gravity"].values

        std_val = np.std(signal)
        peak_to_peak = np.max(signal) - np.min(signal)

        jerk = np.diff(signal)
        jerk_std = np.std(jerk)

        motion_score = std_val + peak_to_peak + jerk_std

        window_infos.append({
            "start": start,
            "motion_score": motion_score
        })

    if len(window_infos) == 0:
        return pd.DataFrame()

    motion_scores = [w["motion_score"] for w in window_infos]
    stop_threshold = np.percentile(motion_scores, STOP_PERCENTILE)

    for w in window_infos:
        start = w["start"]
        motion_score = w["motion_score"]

        if motion_score > stop_threshold:
            continue

        window = df.iloc[start:start + WINDOW_SIZE]
        raw_signal = window["vibration_no_gravity"].values

        try:
            filtered_signal = highpass_filter(
                raw_signal,
                cutoff_freq=LOW_FREQ_CUTOFF,
                sampling_rate=SAMPLING_RATE
            )
        except Exception:
            continue

        mean_val = np.mean(filtered_signal)
        std_val = np.std(filtered_signal)
        rms_val = np.sqrt(np.mean(filtered_signal**2))
        max_val = np.max(filtered_signal)
        min_val = np.min(filtered_signal)
        peak_to_peak = max_val - min_val

        jerk = np.diff(filtered_signal)
        jerk_mean = np.mean(np.abs(jerk))
        jerk_std = np.std(jerk)

        zero_crossings = np.where(np.diff(np.sign(filtered_signal)))[0]
        zero_crossing_rate = len(zero_crossings) / len(filtered_signal)

        fft_values = np.abs(rfft(filtered_signal))
        fft_freqs = rfftfreq(len(filtered_signal), d=1 / SAMPLING_RATE)

        fft_values[0] = 0

        freq_mask = (fft_freqs >= FFT_MIN_FREQ) & (fft_freqs <= FFT_MAX_FREQ)

        selected_freqs = fft_freqs[freq_mask]
        selected_fft = fft_values[freq_mask]

        if len(selected_fft) == 0:
            continue

        peak_index = np.argmax(selected_fft)

        fft_peak_freq = selected_freqs[peak_index]
        fft_peak_mag = selected_fft[peak_index]
        fft_energy = np.sum(selected_fft**2)

        if np.sum(selected_fft) > 0:
            spectral_centroid = np.sum(selected_freqs * selected_fft) / np.sum(selected_fft)
        else:
            spectral_centroid = 0

        fft_prob = selected_fft / (np.sum(selected_fft) + 1e-12)
        spectral_entropy = -np.sum(fft_prob * np.log2(fft_prob + 1e-12))

        features.append({
            "mean": mean_val,
            "std": std_val,
            "rms": rms_val,
            "max": max_val,
            "min": min_val,
            "peak_to_peak": peak_to_peak,
            "jerk_mean": jerk_mean,
            "jerk_std": jerk_std,
            "zero_crossing_rate": zero_crossing_rate,
            "fft_peak_freq": fft_peak_freq,
            "fft_peak_mag": fft_peak_mag,
            "fft_energy": fft_energy,
            "spectral_centroid": spectral_centroid,
            "spectral_entropy": spectral_entropy,
            "motion_score": motion_score
        })

    return pd.DataFrame(features)


# ============================================================
# STREAMLIT APP
# ============================================================

st.set_page_config(
    page_title="EV vs Non-EV Detector",
    page_icon="🚗",
    layout="wide"
)

st.title("EV vs Non-EV Detection using Accelerometer Data")

st.write(
    "Upload an accelerometer file. The app will preprocess the signal, extract FFT-based features, "
    "and predict whether the vehicle is EV-like or Non-EV-like."
)

uploaded_file = st.file_uploader(
    "Upload CSV or TXT file",
    type=["csv", "txt"]
)

if uploaded_file is not None:
    try:
        file_name = uploaded_file.name.lower()

        if file_name.endswith(".csv"):
            original_df = pd.read_csv(uploaded_file)
            cleaned_df = clean_csv_file(original_df)

        elif file_name.endswith(".txt"):
            cleaned_df = clean_txt_file(uploaded_file)
            original_df = cleaned_df.copy()

        else:
            raise ValueError("Unsupported file type.")

        cleaned_df = resample_to_100hz(cleaned_df)

        st.subheader("Cleaned 100 Hz Data Preview")
        st.dataframe(cleaned_df.head())

        st.write("Total rows after cleaning:", len(cleaned_df))
        st.write("Approx duration in seconds:", round(cleaned_df["time_sec"].iloc[-1], 2))

        features_df = extract_features(cleaned_df)

        if features_df.empty:
            st.error("No valid stopped windows found. Upload a longer recording.")
        else:
            X = features_df.reindex(columns=feature_columns, fill_value=0)

            predictions = model.predict(X)

            features_df["prediction"] = predictions

            final_prediction = pd.Series(predictions).value_counts().idxmax()

            ev_count = (features_df["prediction"] == "ev").sum()
            non_ev_count = (features_df["prediction"] == "non_ev").sum()
            total_windows = len(features_df)

            st.subheader("Final Prediction")

            if final_prediction == "ev":
                st.success("Prediction: EV-like")
            else:
                st.warning("Prediction: Non-EV-like")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("EV-like windows", ev_count)

            with col2:
                st.metric("Non-EV-like windows", non_ev_count)

            with col3:
                st.metric("Stopped windows used", total_windows)

            st.subheader("Window-Level Predictions")
            st.dataframe(features_df)

            csv_output = features_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download prediction results",
                data=csv_output,
                file_name="prediction_results.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error: {e}")
