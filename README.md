# MET – Metabolic Equivalent of Task AI Android App

An Android application and machine learning pipeline for **real-time physical activity recognition**.  
The app predicts a user’s current **MET class** (Metabolic Equivalent of Task) from smartphone accelerometer and gyroscope data, and shows **cumulative daily activity** across:

- **Sedentary** (< 1.5 METs)  
- **Light** (1.5–3 METs)  
- **Moderate** (3–6 METs)  
- **Vigorous** (> 6 METs)  

---

## 📱 Android App

- Source code lives in: [`mobile_app/`](mobile_app)  
- Built in Android Studio (Kotlin).  
- Uses **foreground service** for continuous sensing.  
- **On-device inference** with ONNX Runtime.  
- **Room database** stores daily history.  
- Live UI: current MET class + pie chart + history view.  

To build the APK:

1. Open `mobile_app/` in Android Studio.  
2. Go to **Build → Build Bundle(s)/APK(s) → Build APK(s)**.  
3. The APK will appear in `mobile_app/app/build/outputs/apk/`.  

---

## 🧠 Machine Learning Model

- Hybrid model:  
  - **1D CNN** → raw windows `[150 × 8]` (3s @ 50Hz).  
  - **MLP** → 36 engineered features (mean, std, min, max, jerk).  
- Datasets: **WISDM, MotionSense, UCI HAR**.  
- Augmentations: random rotation, time warping.  
- Exported to **ONNX** (see `Models/onnx/`).  

---

## 📊 Evaluation

Results from 5 independent runs on a 20% test set are stored in [`evaluation_results.txt`](evaluation_results.txt).

**Averaged Results:**

| Metric    | Mean ± Std    |
|-----------|---------------|
| Accuracy  | 95.68% ± 0.90 |
| Precision | 95.77% ± 0.77 |
| Recall    | 95.68% ± 0.90 |
| F1-score  | 95.68% ± 0.87 |

---

## 🚀 Training Pipeline

- Code: [`train_eval_met_pipeline.py`](train_eval_met_pipeline.py)  
- Dependencies: [`requirements.txt`](requirements.txt)  

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run training / evaluation
```
# Evaluation mode: 5 runs, averages metrics, saves results
EVAL_MODE=True python train_eval_met_pipeline.py

# Normal training mode: trains once and saves ONNX model
EVAL_MODE=False python train_eval_met_pipeline.py
```

Models and scalers are saved into `Models/`.

---

## 📂 Repository Structure

```
.
├── .vscode/                     # VSCode settings (optional)
├── Models/                      # Trained models, scalers, ONNX
├── data/                        # Placeholder for datasets (WISDM, MotionSense, UCI HAR)
├── mobile_app/                  # Android Studio project
├── evaluation_results.txt       # Evaluation metrics (5 runs averaged)
├── requirements.txt             # Python dependencies
├── train_eval_met_pipeline.py   # Training + evaluation pipeline
└── .gitignore                   # Ignore rules
```

---

## 📄 Deliverables

* ✅ Android app source (`mobile_app/`)
* ✅ APK (buildable via Android Studio)
* ✅ Trained ONNX model + scalers (`Models/`)
* ✅ Training pipeline (`train_eval_met_pipeline.py`)
* ✅ Evaluation results (`evaluation_results.txt`)
* ✅ Report (to be added in `docs/`)
* ✅ Demo video (to be added in `docs/`)

---

## 📚 Datasets' References

* Kwapisz, J. R., Weiss, G. M., & Moore, S. A. (2010).
  *Activity Recognition using Cell Phone Accelerometers.*
  In *Proceedings of the Fourth International Workshop on Knowledge Discovery from Sensor Data (SensorKDD-10) at KDD-10*, Washington, DC, USA.

* Malekzadeh, M., Clegg, R. G., Cavallaro, A., & Haddadi, H. (2019).
  *Mobile Sensor Data Anonymization.*
  In *Proceedings of the International Conference on Internet of Things Design and Implementation (IoTDI '19)*, Montreal, Quebec, Canada (pp. 49–58). ACM. [https://doi.org/10.1145/3302505.3310068](https://doi.org/10.1145/3302505.3310068)

* Reyes-Ortiz, J. L., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2013).
  *Human Activity Recognition Using Smartphones.*
  UCI Machine Learning Repository. [https://doi.org/10.24432/C54S4K](https://doi.org/10.24432/C54S4K)

---

## ✨ Author

Developed by **Stefanos Konstantinou** for **Challenge 2025 – ADAMMA**.
