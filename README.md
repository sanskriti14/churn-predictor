# End-to-End Customer Churn Predictor

🚀 **Live Interactive Demo:** [View the Live Dashboard Here](https://churn-predictor-001.streamlit.app/)  
📊 **Training Workflow:** [Open Jupyter Notebook](./notebooks/churn_training.ipynb)

An interactive, production-grade Machine Learning system built to forecast customer subscription cancellations based on customer support ticket volume, billing pricing structures, and historical account longevity. This project features a robust ensemble learning pipeline and an interactive web dashboard for real-time risk assessment.

---

## 📊 Model Performance & Metrics

The predictive model was trained using **Scikit-Learn's Random Forest Classifier** on a synthetic dataset of 200 customers, optimizing for high retention safety.

* **Overall Accuracy:** 93%
* **Churn Precision:** 94% (Low false-alarm rate for proactive customer campaigns)
* **Churn Recall:** 98% (Successfully flags nearly every single customer at risk)

### Evaluation Confusion Matrix
```text
[[ 8  3]  <- [True Negatives,  False Positives]
 [ 1 48]] <- [False Negatives, True Positives]
```

- True Negatives (8): Correctly predicted stable accounts.
- False Positives (3): Safe accounts flagged as risk variables.
- False Negatives (1): Only one account slipped past the safety net undetected.
- True Positives (48): Successfully captured endangered accounts for proactive retention.


## Tech Stack & Tools
* **Language:** Python 3.12
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning Framework:** Scikit-Learn (Random Forest Classifier)
* **Application Infrastructure:** Streamlit


## Core Project Pipeline Architecture
1. **Inference Pipelines:** Loads raw business metrics data, handles missing configurations, and maps categorical text attributes to scalar input matrices.
2. **Modeling Optimization:** Implements a multi-decision tree ensemble algorithm (`RandomForestClassifier`) to predict discrete binary retention classification outputs.
3. **Serving Layer:** Exposes model prediction vectors instantly via an interactive local UI frontend canvas without external JavaScript requirements.


## Local Execution Instructions

### 1. Installation Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Synthetic Customer Dataset (Optional)
```bash
python src/generate_data.py
```

### 3. Execution of Modeling Core
```bash
python src/train.py
```

### 4. Initialize Interactive Canvas Frontend Application
```bash
streamlit run src/app.py
```
