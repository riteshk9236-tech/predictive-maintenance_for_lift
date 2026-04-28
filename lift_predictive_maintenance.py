"""
Lift Predictive Maintenance — Real-Time Sensing & Prediction
=============================================================
Pipeline:
  1. Simulate 3 sensors (revolutions, vibration, humidity) at 4 Hz
  2. Apply multi-layer digital filters (Butterworth LP → Moving Avg → Median)
  3. Collect & store 8 hours of filtered data to CSV
  4. Engineer features to match the pre-trained model
  5. Load pre-trained model → predict per sample → 1 final verdict

Uses existing model: saved_models_no_x_sensors/best_model.joblib
"""
import os, sys, time, warnings, subprocess
from datetime import datetime
import numpy as np
import pandas as pd

try:
    from scipy.signal import butter, lfilter, medfilt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    from scipy.signal import butter, lfilter, medfilt

try:
    import joblib
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])
    import joblib

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
SAMPLE_RATE_HZ     = 4
DEMO_MODE          = True
DEMO_DURATION_SEC  = 30
DEMO_TOTAL_SAMPLES = SAMPLE_RATE_HZ * DEMO_DURATION_SEC
TOTAL_SAMPLES_8HR  = SAMPLE_RATE_HZ * 8 * 3600  # 115,200
FILTER_BATCH_SIZE  = 20
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH         = os.path.join(BASE_DIR, "saved_models_no_x_sensors", "best_model.joblib")
BUTTER_ORDER       = 4
BUTTER_CUTOFF_HZ   = 0.8
MA_WINDOW          = 5
MEDIAN_KERNEL      = 3

# ═══════════════════════════════════════════════════════════════════════════
# SENSOR SIMULATION
# ═══════════════════════════════════════════════════════════════════════════
class SensorSimulator:
    def __init__(self, sample_rate=SAMPLE_RATE_HZ):
        self._t  = 0.0
        self._dt = 1.0 / sample_rate
    def read(self):
        t = self._t
        revolutions = 60.0 + 5.0*np.sin(2*np.pi*0.002*t) + np.random.normal(0, 0.5)
        vibration   = 15.0 + 3.0*np.sin(2*np.pi*0.01*t)  + np.random.normal(0, 1.0)
        if np.random.rand() < 0.005:
            vibration += np.random.uniform(15, 30)
        humidity = 74.0 + 0.5*np.sin(2*np.pi*0.0005*t) + np.random.normal(0, 0.1)
        self._t += self._dt
        return {"revolutions": round(revolutions,3),
                "vibration":   round(vibration,3),
                "humidity":    round(humidity,3)}

# ═══════════════════════════════════════════════════════════════════════════
# DIGITAL FILTER PIPELINE (3 layers)
# ═══════════════════════════════════════════════════════════════════════════
class DigitalFilterPipeline:
    """Layer 1: Butterworth LP | Layer 2: Moving Avg | Layer 3: Median"""
    def __init__(self, fs=SAMPLE_RATE_HZ, butter_order=BUTTER_ORDER,
                 butter_cutoff=BUTTER_CUTOFF_HZ, ma_window=MA_WINDOW,
                 med_kernel=MEDIAN_KERNEL):
        self.ma_window  = ma_window
        self.med_kernel = med_kernel
        nyq = 0.5 * fs
        norm_cutoff = min(max(butter_cutoff / nyq, 0.01), 0.99)
        self.b, self.a = butter(butter_order, norm_cutoff, btype="low")
    def apply(self, signal):
        if len(signal) < max(self.ma_window, self.med_kernel, 6):
            return signal
        filtered = lfilter(self.b, self.a, signal)
        kernel = np.ones(self.ma_window) / self.ma_window
        filtered = np.convolve(filtered, kernel, mode="same")
        filtered = medfilt(filtered, kernel_size=self.med_kernel)
        return filtered

# ═══════════════════════════════════════════════════════════════════════════
# 8-HOUR DATA COLLECTION & STORAGE
# ═══════════════════════════════════════════════════════════════════════════
def collect_and_store_data(demo=DEMO_MODE):
    total = DEMO_TOTAL_SAMPLES if demo else TOTAL_SAMPLES_8HR
    label = f"{DEMO_DURATION_SEC}s (demo)" if demo else "8 hours"
    print("\n" + "=" * 65)
    print(f"  SENSING — Collecting & filtering data  ({label})")
    print(f"  Rate: {SAMPLE_RATE_HZ} Hz  |  Samples: {total:,}")
    print("=" * 65)
    sensor   = SensorSimulator()
    pipeline = DigitalFilterPipeline()
    raw_rev, raw_vib, raw_hum = [], [], []
    filt_rev, filt_vib, filt_hum = [], [], []
    timestamps = []
    t0 = time.time()
    for i in range(total):
        r = sensor.read()
        raw_rev.append(r["revolutions"])
        raw_vib.append(r["vibration"])
        raw_hum.append(r["humidity"])
        timestamps.append(datetime.now().isoformat())
        if (i+1) % FILTER_BATCH_SIZE == 0:
            filt_rev.extend(pipeline.apply(np.array(raw_rev[-FILTER_BATCH_SIZE:])).tolist())
            filt_vib.extend(pipeline.apply(np.array(raw_vib[-FILTER_BATCH_SIZE:])).tolist())
            filt_hum.extend(pipeline.apply(np.array(raw_hum[-FILTER_BATCH_SIZE:])).tolist())
        if (i+1) % (total//5) == 0:
            print(f"    ▸ {(i+1)/total*100:5.1f}%  ({i+1:,} samples, {time.time()-t0:.1f}s)")
    rem = total % FILTER_BATCH_SIZE
    if rem > 0:
        filt_rev.extend(pipeline.apply(np.array(raw_rev[-rem:])).tolist())
        filt_vib.extend(pipeline.apply(np.array(raw_vib[-rem:])).tolist())
        filt_hum.extend(pipeline.apply(np.array(raw_hum[-rem:])).tolist())
    n = min(len(filt_rev), len(filt_vib), len(filt_hum), len(timestamps))
    csv_path = os.path.join(BASE_DIR,
                            f"lift_sensor_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df = pd.DataFrame({"timestamp": timestamps[:n],
                        "revolutions": filt_rev[:n],
                        "vibration": filt_vib[:n],
                        "humidity": filt_hum[:n]})
    df.to_csv(csv_path, index=False)
    print(f"\n    ✅ {len(df):,} filtered samples saved → {csv_path}")
    return csv_path

# ═══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — match what the pre-trained model expects
# ═══════════════════════════════════════════════════════════════════════════
def engineer_features(df):
    """
    Engineer the exact features the pre-trained model expects:
      rev_hum_product, rev_hum_ratio, rolling_vibration_std, revolutions_sq,
      rolling_vibration_mean, vibration_diff, vibration_lag_3, ema_vibration,
      vibration_lag_2, vibration_lag_1, humidity_sq
    Plus the 3 raw features: revolutions, humidity, vibration.
    """
    out = pd.DataFrame()
    out["revolutions"] = df["revolutions"]
    out["humidity"]    = df["humidity"]
    out["vibration"]   = df["vibration"]

    # Engineered features (matching pre-trained model)
    out["rev_hum_product"]       = df["revolutions"] * df["humidity"]
    out["rev_hum_ratio"]         = df["revolutions"] / (df["humidity"] + 1e-9)
    out["revolutions_sq"]        = df["revolutions"] ** 2
    out["humidity_sq"]           = df["humidity"] ** 2
    out["vibration_diff"]        = df["vibration"].diff().fillna(0)
    out["vibration_lag_1"]       = df["vibration"].shift(1).fillna(method="bfill")
    out["vibration_lag_2"]       = df["vibration"].shift(2).fillna(method="bfill")
    out["vibration_lag_3"]       = df["vibration"].shift(3).fillna(method="bfill")
    out["rolling_vibration_mean"] = df["vibration"].rolling(window=10, min_periods=1).mean()
    out["rolling_vibration_std"]  = df["vibration"].rolling(window=10, min_periods=1).std().fillna(0)
    out["ema_vibration"]          = df["vibration"].ewm(span=10, min_periods=1).mean()

    return out

# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
def predict_from_csv(csv_path):
    """
    Load 8-hour CSV → engineer features → run through pre-trained model
    → aggregate into 1 final maintenance decision.
    """
    print("\n" + "=" * 65)
    print("  PREDICTION — Using Pre-Trained Model")
    print("=" * 65)
    if not os.path.exists(MODEL_PATH):
        print(f"    ❌ Model not found: {MODEL_PATH}")
        sys.exit(1)
    clf = joblib.load(MODEL_PATH)
    print(f"    Model loaded: {MODEL_PATH}")

    df = pd.read_csv(csv_path)
    print(f"    Data loaded : {len(df):,} samples")

    # Engineer features to match model
    print("    Engineering features ...")
    X = engineer_features(df)
    print(f"    Features: {list(X.columns)}")

    # Per-sample prediction
    print("    Running predictions ...")
    y_pred = clf.predict(X)

    total       = len(y_pred)
    maint_count = int(np.sum(y_pred == 1))
    ok_count    = int(np.sum(y_pred == 0))
    maint_pct   = maint_count / total * 100

    print(f"    OK samples          : {ok_count:,} ({100-maint_pct:.2f}%)")
    print(f"    MAINTENANCE samples : {maint_count:,} ({maint_pct:.2f}%)")

    # Probabilities
    try:
        proba = clf.predict_proba(X)
        avg_prob = float(np.mean(proba[:, 1])) if proba.shape[1] > 1 else 0.0
    except Exception:
        avg_prob = maint_pct / 100.0

    # Final decision: >1% flagged → maintenance
    THRESHOLD = 1.0
    needs_maint = maint_pct > THRESHOLD
    verdict = "⚠️  MAINTENANCE REQUIRED" if needs_maint \
              else "✅  LIFT OK — No Maintenance Needed"

    print()
    print(f"    ╔{'═'*55}╗")
    print(f"    ║{'FINAL 8-HOUR PREDICTION':^55s}║")
    print(f"    ╠{'═'*55}╣")
    print(f"    ║  Result             : {verdict:<30s}║")
    print(f"    ║  Flagged samples    : {maint_count:>6,} / {total:<6,}{' '*17}║")
    print(f"    ║  Maintenance %      : {maint_pct:>6.2f}%{' '*26}║")
    print(f"    ║  Avg maint. prob    : {avg_prob*100:>6.2f}%{' '*26}║")
    print(f"    ║  Threshold          : {THRESHOLD:>6.2f}%{' '*26}║")
    print(f"    ╚{'═'*55}╝")
    return {"prediction": 1 if needs_maint else 0, "label": verdict}

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print()
    print("╔" + "═"*63 + "╗")
    print("║  LIFT PREDICTIVE MAINTENANCE — REAL-TIME SYSTEM               ║")
    print("║  3 Sensors × 4Hz → Digital Filters → 8hr Store → Predict     ║")
    print("╚" + "═"*63 + "╝")
    print(f"  Mode  : {'DEMO' if DEMO_MODE else 'PRODUCTION (8 hours)'}")
    print(f"  Model : {MODEL_PATH}")

    # Step 1: Sense 8 hours of data with digital filtering
    csv_path = collect_and_store_data(demo=DEMO_MODE)

    # Step 2: Load pre-trained model, engineer features, predict
    result = predict_from_csv(csv_path)

    # Summary
    print("\n" + "=" * 65)
    print(f"  VERDICT: {result['label']}")
    print("=" * 65)
    print("  Done! 🎉\n")

if __name__ == "__main__":
    main()