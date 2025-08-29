# train_met_pipeline copy.py
# Hybrid HAR model: 1D-CNN on raw (acc+gyro+magnitudes) + small MLP on engineered features
# Data: WISDM (acc only, gyro=0), MotionSense (acc+gyro), UCI HAR raw (acc+gyro)
# Exports: models/hybrid_saved_model  and  Models/onnx/hybrid_met.onnx
# -----------------------------------

import os, glob, sys, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

# ------------------ Config ------------------
WISDM_ROOT       = "data/WISDM_ar_v1.1"
MOTIONSENSE_ROOT = "data/motion-sense"               # your repo layout (dws_1/, jog_1/, ...)
UCIHAR_RAW_ROOT  = "data/UCI-HAR-raw"                # we will append "UCI HAR Dataset" inside loader

TARGET_HZ  = 50
WIN_SEC    = 3.0
STEP_SEC   = 1.5

# --- Evaluation Mode Toggle ---
# Set to True to run evaluation: trains the model 5 times on a train/test split,
# averages the results, and saves them to a text file without saving the model.
# Set to False for normal training: trains once on all data and saves the model.
EVAL_MODE = False
RUNS = 5
RESULTS_FILE = "evaluation_results.txt"
# -----------------------------

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# 0 Sed, 1 Light, 2 Moderate, 3 Vig
MET_MAP = {
    "Sitting": 0, "Standing": 0, "Laying": 0,
    "Walking": 1,
    "Upstairs": 2, "Downstairs": 2,
    "Jogging": 3, "Running": 3,
    "sit": 0, "std": 0, "wlk": 1, "ups": 2, "dws": 2, "jog": 3
}

# ------------------ Utils ------------------
def to_seconds(ts):
    ts = np.asarray(ts).astype("float64")
    if np.nanmax(ts) > 1e10:  # ns or ms
        ts = ts / 1000.0
    return ts

def resample_time_series(df, hz=TARGET_HZ):
    df = df.sort_values("timestamp")
    if df.empty: 
        return df
    t0, t1 = df["timestamp"].iloc[0], df["timestamp"].iloc[-1]
    if t1 <= t0:
        return pd.DataFrame(columns=df.columns)
    new_t = np.arange(t0, t1, 1.0/hz)
    out = pd.DataFrame({"timestamp": new_t})
    for col in ["x","y","z","gx","gy","gz"]:
        if col in df.columns:
            out[col] = np.interp(new_t, df["timestamp"].values, df[col].values)
    return out

def window_iter(n, win, step):
    i = 0
    while i + win <= n:
        yield i, i+win
        i += step

def random_rotation_matrix(max_angle_deg=60):
    ax = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg))
    ay = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg))
    az = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg))
    Rx = np.array([[1,0,0],[0,np.cos(ax),-np.sin(ax)],[0,np.sin(ax),np.cos(ax)]])
    Ry = np.array([[np.cos(ay),0,np.sin(ay)],[0,1,0],[-np.sin(ay),0,np.cos(ay)]])
    Rz = np.array([[np.cos(az),-np.sin(az),0],[np.sin(az),np.cos(az),0],[0,0,1]])
    return Rz @ Ry @ Rx

def time_warp(ts, factor_range=(0.9, 1.1)):
    f = np.random.uniform(*factor_range)
    c = (ts[0] + ts[-1]) / 2.0
    return c + (ts - c) * f

def svm(x,y,z):
    return np.sqrt(x*x + y*y + z*z)

# ------------------ Loaders ------------------
def load_wisdm(root, frac=0.4):
    """
    Load WISDM v1.1 dataset (accelerometer only).
    Limit to `frac` of rows to avoid OOM.
    """
    import glob, os
    import pandas as pd
    import numpy as np

    raw_files = [f for f in glob.glob(os.path.join(root, "**/*raw*.txt"), recursive=True)
                 if "about" not in os.path.basename(f).lower()]
    if not raw_files:
        print("No WISDM raw data file found in:", root)
        return pd.DataFrame()

    path = raw_files[0]
    print("Loading WISDM from:", path)

    cols = ["user","activity","timestamp","x","y","z"]
    df = pd.read_csv(path, header=None, names=cols, on_bad_lines="skip")

    # Clean z
    if df["z"].dtype == object:
        df["z"] = df["z"].str.replace(";", "", regex=False).astype(float)

    # Convert timestamp: nanoseconds → seconds
    df["timestamp"] = df["timestamp"].astype(np.float64) / 1e9

    # Add gyro placeholders
    df["gx"] = 0.0
    df["gy"] = 0.0
    df["gz"] = 0.0
    df["dataset"] = "WISDM"

    # Sample only a fraction
    if frac < 1.0:
        df = df.sample(frac=frac, random_state=42)

    print("Loaded WISDM shape:", df.shape)
    return df




def load_motionsense(root):
    if not os.path.exists(root):
        return pd.DataFrame()
    labels = ["dws","ups","wlk","jog","sit","std"]
    rows = []
    for lab in labels:
        for trial_dir in sorted(glob.glob(os.path.join(root, f"{lab}_*"))):
            for csv in sorted(glob.glob(os.path.join(trial_dir, "sub_*.csv"))):
                sub = os.path.basename(csv).split("_")[1].split(".")[0]
                df = pd.read_csv(csv)
                if "Unnamed: 0" in df.columns:
                    df = df.drop(columns=["Unnamed: 0"])
                n = len(df)
                if n == 0: continue
                ts = np.arange(n) / 50.0
                rows.append(pd.DataFrame({
                    "user": sub,
                    "activity": lab,
                    "timestamp": ts,
                    "x": df["userAcceleration.x"].values,
                    "y": df["userAcceleration.y"].values,
                    "z": df["userAcceleration.z"].values,
                    "gx": df["rotationRate.x"].values,
                    "gy": df["rotationRate.y"].values,
                    "gz": df["rotationRate.z"].values,
                    "dataset": "MotionSense"
                }))
    output_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    print("Loaded MotionSense shape:", output_df.shape)
    return output_df


def load_ucihar_raw(root):
    dataset_dir = os.path.join(root, "UCI HAR Dataset")
    if not os.path.exists(dataset_dir): 
        return pd.DataFrame()

    def load_split(split):
        split_dir = os.path.join(dataset_dir, split, "Inertial Signals")
        ax = np.loadtxt(os.path.join(split_dir, f"total_acc_x_{split}.txt"))
        ay = np.loadtxt(os.path.join(split_dir, f"total_acc_y_{split}.txt"))
        az = np.loadtxt(os.path.join(split_dir, f"total_acc_z_{split}.txt"))
        gx = np.loadtxt(os.path.join(split_dir, f"body_gyro_x_{split}.txt"))
        gy = np.loadtxt(os.path.join(split_dir, f"body_gyro_y_{split}.txt"))
        gz = np.loadtxt(os.path.join(split_dir, f"body_gyro_z_{split}.txt"))
        y  = np.loadtxt(os.path.join(dataset_dir, split, f"y_{split}.txt")).astype(int)
        sb = np.loadtxt(os.path.join(dataset_dir, split, f"subject_{split}.txt")).astype(int)

        rows = []
        nwin, winlen = ax.shape
        t = np.arange(winlen)/50.0
        id2act = {1:"Walking",2:"Upstairs",3:"Downstairs",4:"Sitting",5:"Standing",6:"Laying"}
        for i in range(nwin):
            rows.append(pd.DataFrame({
                "user": int(sb[i]),
                "activity": id2act[int(y[i])],
                "timestamp": t,
                "x": ax[i], "y": ay[i], "z": az[i],
                "gx": gx[i], "gy": gy[i], "gz": gz[i],
                "dataset": "UCIHAR"
            }))
        return pd.concat(rows, ignore_index=True)

    tr = load_split("train")
    te = load_split("test")
    output_df = pd.concat([tr, te], ignore_index=True)
    print("Loaded UCI HAR raw shape:", output_df.shape)
    return output_df
# ------------------ Build combined df + augmentation ------------------
def build_base_df():
    dfs = []

    # Limit WISDM
    w = load_wisdm(WISDM_ROOT)
    if len(w):
        dfs.append(w)

    m = load_motionsense(MOTIONSENSE_ROOT)
    if len(m):
        dfs.append(m)

    u = load_ucihar_raw(UCIHAR_RAW_ROOT)
    if len(u):
        dfs.append(u)

    if not dfs:
        print("No dataset found.")
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)
    df["activity"] = df["activity"].replace(
        {"dws":"Downstairs","ups":"Upstairs","wlk":"Walking","jog":"Jogging","sit":"Sitting","std":"Standing"}
    )
    df["met_class"] = df["activity"].map(MET_MAP)
    df = df.dropna(subset=["met_class"])

    out_rows = []
    for (dataset, user, act), g in df.groupby(["dataset","user","activity"]):
        g = g.sort_values("timestamp")

        if dataset == "WISDM":
            hz = 20   # skip resample, keep native
            g["hz"] = hz
            out_rows.append(g)
        else:
            # Resample to TARGET_HZ
            rs = resample_time_series(g[["timestamp","x","y","z","gx","gy","gz"]], hz=TARGET_HZ)
            rs["user"] = str(user)
            rs["activity"] = act
            rs["dataset"] = dataset
            rs["hz"] = TARGET_HZ
            out_rows.append(rs)

    full = pd.concat(out_rows, ignore_index=True)
    full["met_class"] = full["activity"].map(MET_MAP)
    full = full.dropna(subset=["met_class"])
    return full


def augment_windows(Xraw, Xfeat, y, prob=0.3):
    Xr, Xf, Y = [], [], []
    for i in range(len(Xraw)):
        Xr.append(Xraw[i]); Xf.append(Xfeat[i]); Y.append(y[i])
        if np.random.rand() < prob:
            # Random rotation
            R = random_rotation_matrix(60)
            raw = Xraw[i].copy()  # [win, 8]
            # Rotate accel + gyro separately
            acc = raw[:,0:3].T  # [3,win]
            gyr = raw[:,3:6].T
            acc_r = (R @ acc).T
            gyr_r = (R @ gyr).T
            raw[:,0:3] = acc_r
            raw[:,3:6] = gyr_r
            Xr.append(raw)
            Xf.append(Xfeat[i])
            Y.append(y[i])
    return np.stack(Xr), np.stack(Xf), np.array(Y)


def augment_rotate_time_all(df, hz, prob=0.8):
    # rotate both accel and gyro; light time warp; resample back to hz
    df = df.sample(frac=0.2, random_state=42) # augment only 20% of data
    rows = []
    for (u,a), g in df.groupby(["user","activity"]):
        g = g.sort_values("timestamp").copy()
        if len(g) < int(WIN_SEC*hz): 
            rows.append(g); continue

        if np.random.rand() < prob:
            R = random_rotation_matrix(60)
            acc = np.stack([g["x"].values, g["y"].values, g["z"].values], axis=0)
            gyr = np.stack([g["gx"].values, g["gy"].values, g["gz"].values], axis=0)
            acc_r = (R @ acc).T
            gyr_r = (R @ gyr).T
            g["x"], g["y"], g["z"]   = acc_r[:,0], acc_r[:,1], acc_r[:,2]
            g["gx"], g["gy"], g["gz"] = gyr_r[:,0], gyr_r[:,1], gyr_r[:,2]
            tw = time_warp(g["timestamp"].values, (0.9, 1.1))
            g["timestamp"] = tw
            g = resample_time_series(g[["timestamp","x","y","z","gx","gy","gz"]], hz=hz)
            g["user"] = u; g["activity"] = a

        rows.append(g)
    out = pd.concat(rows, ignore_index=True)
    out["met_class"] = out["activity"].map(MET_MAP)
    return out

# ------------------ Featurization ------------------
def basic_stats(arr):
    s = pd.Series(arr)
    return [float(s.mean()), float(s.std(ddof=0)), float(s.min()), float(s.max())]

def extract_features(win_df):
    # 32 + 4 jerk = 36 features (orientation-invariant magnitudes included)
    x = win_df["x"].values;  y = win_df["y"].values;  z = win_df["z"].values
    gx = win_df["gx"].values; gy = win_df["gy"].values; gz = win_df["gz"].values
    amag  = svm(x,y,z)
    gmag  = svm(gx,gy,gz)
    # jerk on accel magnitude (captures dynamics, still orientation-invariant)
    jerk = np.diff(amag, prepend=amag[0]) * TARGET_HZ

    feats = []
    for a in [x,y,z,gx,gy,gz, amag, gmag]:
        feats += basic_stats(a)
    feats += basic_stats(jerk)
    return np.array(feats, dtype=np.float32)  # 36-dim

def make_windows(df):
    win = int(WIN_SEC*TARGET_HZ)
    step = int(STEP_SEC*TARGET_HZ)
    Xraw, Xfeat, y = [], [], []
    for (u,a), g in df.groupby(["user","activity"]):
        g = g.reset_index(drop=True)
        for a_i, b_i in window_iter(len(g), win, step):
            w = g.iloc[a_i:b_i]
            if len(w) < win: continue

            # raw channels + magnitudes -> [win, 8]
            x = w[["x","y","z","gx","gy","gz"]].values.astype("float32")
            amag = np.sqrt((x[:,0]**2 + x[:,1]**2 + x[:,2]**2)).astype("float32")[:,None]
            gmag = np.sqrt((x[:,3]**2 + x[:,4]**2 + x[:,5]**2)).astype("float32")[:,None]
            raw8 = np.concatenate([x, amag, gmag], axis=1)  # [win, 8]
            Xraw.append(raw8)

            # 36-d engineered features
            Xfeat.append(extract_features(w))
            y.append(int(MET_MAP[w["activity"].iloc[0]]))
    return np.stack(Xraw), np.stack(Xfeat), np.array(y, dtype=np.int64)



from sklearn.preprocessing import StandardScaler
# Note: numpy is already imported at the top of the file

def scale_inputs(Xraw, Xfeat):
    """
    Scales raw windows and feature arrays with StandardScaler.
    Cleans NaNs/Infs to avoid NaN loss during training.
    
    Args:
        Xraw: [N, win, 8]
        Xfeat: [N, 36]
    Returns:
        Xraw_scaled, Xfeat_scaled, raw_scaler, feat_scaler
    """
    # --- Flatten raw for scaling ---
    N, win, C = Xraw.shape
    Xraw_flat = Xraw.reshape((-1, C))

    raw_scaler = StandardScaler()
    Xraw_scaled = raw_scaler.fit_transform(Xraw_flat)
    Xraw_scaled = Xraw_scaled.reshape(N, win, C)

    feat_scaler = StandardScaler()
    Xfeat_scaled = feat_scaler.fit_transform(Xfeat)

    # --- Clean NaNs/Infs ---
    Xraw_scaled = np.nan_to_num(Xraw_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    Xfeat_scaled = np.nan_to_num(Xfeat_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    print("Scaled Xraw:", Xraw_scaled.shape,
          "min:", Xraw_scaled.min(), "max:", Xraw_scaled.max())
    print("Scaled Xfeat:", Xfeat_scaled.shape,
          "min:", Xfeat_scaled.min(), "max:", Xfeat_scaled.max())

    return Xraw_scaled, Xfeat_scaled, raw_scaler, feat_scaler


# ------------------ Train hybrid model ------------------
def train_hybrid(Xraw, Xfeat, y):
    import tensorflow as tf
    from tensorflow import keras

    # We set the seed for each run in main(), so no need to set it here
    
    win = Xraw.shape[1]

    # ---- Inputs ----
    raw_in  = keras.layers.Input(shape=(win, 8), name="raw_input")
    feat_in = keras.layers.Input(shape=(Xfeat.shape[1],), name="feat_input")

    # ---- CNN branch ----
    x = keras.layers.Conv1D(32, 5, padding="same")(raw_in)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv1D(64, 5, strides=2, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.GlobalAveragePooling1D()(x)

    # ---- Feature branch ----
    f = keras.layers.Dense(64, activation="relu")(feat_in)
    f = keras.layers.Dropout(0.3)(f)
    f = keras.layers.Dense(32, activation="relu")(f)

    # ---- Fusion ----
    h = keras.layers.Concatenate()([x, f])
    h = keras.layers.Dense(64, activation="relu")(h)
    h = keras.layers.Dropout(0.3)(h)
    out = keras.layers.Dense(4, activation="softmax", name="probs")(h)

    model = keras.Model(inputs=[raw_in, feat_in], outputs=out)

    # ---- Compile ----
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # ---- Callbacks ----
    cb = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]

    # ---- Train ----
    model.fit(
        [Xraw, Xfeat], y,
        epochs=30,
        batch_size=256,
        validation_split=0.1,
        callbacks=cb,
        verbose=2
    )

    return model


# ------------------ Export to ONNX ------------------
def export_to_onnx(saved_model_dir, onnx_path):
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    # Use CLI to avoid API version quirks
    import subprocess, sys as _sys
    cmd = [
        _sys.executable, "-m", "tf2onnx.convert",
        "--saved-model", saved_model_dir,
        "--output", onnx_path,
        "--opset", "13"
    ]
    print("Converting to ONNX:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError("tf2onnx conversion failed")
    print("Saved ONNX:", onnx_path)

# ------------------ Main ------------------
def main():
    print("Building base dataframe...")
    base = build_base_df()
    print(base.groupby(["dataset","activity"]).size().reset_index(name="n"))

    print("Windowing & features...")
    Xraw, Xfeat, y = make_windows(base)
    print("Raw:", Xraw.shape, "Feat:", Xfeat.shape, "y:", y.shape)

    print("Augmenting at window level...")
    Xraw_aug, Xfeat_aug, y_aug = augment_windows(Xraw, Xfeat, y, prob=0.3)
    print("After augmentation:", Xraw_aug.shape, Xfeat_aug.shape, y_aug.shape)


    if EVAL_MODE:
        # ==================================
        #        EVALUATION MODE
        # ==================================
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        accuracies, f1_scores, precisions, recalls = [], [], [], []
        print(f"\n--- Starting Evaluation Mode ({RUNS} runs) ---")

        # Split data into train and test sets once, before the loop
        Xraw_train, Xraw_test, Xfeat_train, Xfeat_test, y_train, y_test = train_test_split(
            Xraw_aug, Xfeat_aug, y_aug, test_size=0.2, random_state=RANDOM_SEED, stratify=y_aug
        )

        for i in range(RUNS):
            run_seed = RANDOM_SEED + i
            print(f"\n--- RUN {i+1}/{RUNS} (seed={run_seed}) ---")
            np.random.seed(run_seed)
            tf.random.set_seed(run_seed)

            # --- Scale inputs for this run ---
            raw_scaler = StandardScaler()
            N_tr, win_tr, C_tr = Xraw_train.shape
            Xraw_train_flat = Xraw_train.reshape((-1, C_tr))
            raw_scaler.fit(Xraw_train_flat)

            feat_scaler = StandardScaler()
            feat_scaler.fit(Xfeat_train)
            
            Xraw_train_scaled = raw_scaler.transform(Xraw_train_flat).reshape(N_tr, win_tr, C_tr)
            Xfeat_train_scaled = feat_scaler.transform(Xfeat_train)

            N_te, win_te, C_te = Xraw_test.shape
            Xraw_test_flat = Xraw_test.reshape((-1, C_te))
            Xraw_test_scaled = raw_scaler.transform(Xraw_test_flat).reshape(N_te, win_te, C_te)
            Xfeat_test_scaled = feat_scaler.transform(Xfeat_test)

            Xraw_train_scaled = np.nan_to_num(Xraw_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            Xfeat_train_scaled = np.nan_to_num(Xfeat_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            Xraw_test_scaled = np.nan_to_num(Xraw_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            Xfeat_test_scaled = np.nan_to_num(Xfeat_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            
            model = train_hybrid(Xraw_train_scaled, Xfeat_train_scaled, y_train)

            print("Evaluating on test set...")
            y_pred_probs = model.predict([Xraw_test_scaled, Xfeat_test_scaled], verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)
            
            report = classification_report(y_test, y_pred, output_dict=True)
            acc = report['accuracy']
            f1 = report['weighted avg']['f1-score']
            precision = report['weighted avg']['precision']
            recall = report['weighted avg']['recall']
            
            print(f"Run {i+1} Accuracy: {acc:.4f}, F1: {f1:.4f}")
            accuracies.append(acc)
            f1_scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)

        # --- Summarize and save evaluation results ---
        print("\n--- Evaluation Summary ---")
        print(f"Average Accuracy:  {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")
        print(f"Average Precision: {np.mean(precisions):.4f} +/- {np.std(precisions):.4f}")
        print(f"Average Recall:    {np.mean(recalls):.4f} +/- {np.std(recalls):.4f}")
        print(f"Average F1-Score:  {np.mean(f1_scores):.4f} +/- {np.std(f1_scores):.4f}")

        with open(RESULTS_FILE, "w") as f:
            f.write(f"Evaluation Results ({RUNS} runs on a 20% test set)\n")
            f.write("----------------------------------------------------\n")
            f.write(f"Mean Accuracy:    {np.mean(accuracies):.4f} (std: {np.std(accuracies):.4f})\n")
            f.write(f"Mean Precision:   {np.mean(precisions):.4f} (std: {np.std(precisions):.4f})\n")
            f.write(f"Mean Recall:      {np.mean(recalls):.4f} (std: {np.std(recalls):.4f})\n")
            f.write(f"Mean F1-Score:    {np.mean(f1_scores):.4f} (std: {np.std(f1_scores):.4f})\n\n")
            f.write("Individual Run F1-Scores:\n")
            for i, score in enumerate(f1_scores):
                f.write(f"  Run {i+1}: {score:.4f}\n")
        print(f"\nResults saved to {RESULTS_FILE}")

    else:
        # ==================================
        #        NORMAL TRAINING MODE
        # ==================================
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        print("\nSplitting data into 80% train / 20% test for training and final evaluation...")
        Xraw_train, Xraw_test, Xfeat_train, Xfeat_test, y_train, y_test = train_test_split(
            Xraw_aug, Xfeat_aug, y_aug, test_size=0.2, random_state=RANDOM_SEED, stratify=y_aug
        )

        print("Scaling inputs (fitting on train set only)...")
        raw_scaler = StandardScaler()
        N_tr, win_tr, C_tr = Xraw_train.shape
        Xraw_train_flat = Xraw_train.reshape((-1, C_tr))
        raw_scaler.fit(Xraw_train_flat)
        Xraw_train_scaled = raw_scaler.transform(Xraw_train_flat).reshape(N_tr, win_tr, C_tr)

        feat_scaler = StandardScaler()
        feat_scaler.fit(Xfeat_train)
        Xfeat_train_scaled = feat_scaler.transform(Xfeat_train)

        N_te, win_te, C_te = Xraw_test.shape
        Xraw_test_flat = Xraw_test.reshape((-1, C_te))
        Xraw_test_scaled = raw_scaler.transform(Xraw_test_flat).reshape(N_te, win_te, C_te)
        Xfeat_test_scaled = feat_scaler.transform(Xfeat_test)

        Xraw_train_scaled = np.nan_to_num(Xraw_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        Xfeat_train_scaled = np.nan_to_num(Xfeat_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        Xraw_test_scaled = np.nan_to_num(Xraw_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        Xfeat_test_scaled = np.nan_to_num(Xfeat_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        print("Training hybrid model on the 80% training split...")
        model = train_hybrid(Xraw_train_scaled, Xfeat_train_scaled, y_train)

        print("\n--- Final Model Performance on the 20% Held-Out Test Set ---")
        y_pred_probs = model.predict([Xraw_test_scaled, Xfeat_test_scaled], verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        target_names = ['Class 0 (Sed)', 'Class 1 (Light)', 'Class 2 (Moderate)', 'Class 3 (Vigorous)']
        print(classification_report(y_test, y_pred, target_names=target_names))

        save_choice = input("Save the model, scalers, and ONNX file? (y/n): ").lower().strip()

        if save_choice == 'y':
            print("\nSaving model trained on 80% of data and corresponding scalers...")
            # --- Save scalers ---
            os.makedirs("Models/scalers", exist_ok=True)
            np.save("Models/scalers/raw_means.npy", raw_scaler.mean_.astype(np.float32))
            np.save("Models/scalers/raw_stds.npy",  raw_scaler.scale_.astype(np.float32))
            np.save("Models/scalers/feat_means.npy", feat_scaler.mean_.astype(np.float32))
            np.save("Models/scalers/feat_stds.npy",  feat_scaler.scale_.astype(np.float32))
            print("Saved scaler stats in Models/scalers/")

            # --- Export Keras SavedModel ---
            saved_dir = "models/hybrid_saved_model"
            os.makedirs(os.path.dirname(saved_dir), exist_ok=True)
            model.export(saved_dir)
            print(f"Saved Keras model to: {saved_dir}")

            # --- Export to ONNX ---
            export_to_onnx(saved_dir, "Models/onnx/hybrid_met.onnx")
        else:
            print("Model and assets were not saved.")



if __name__ == "__main__":
    main()