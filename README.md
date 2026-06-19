# End-to-End Customer Churn Predictor

🚀 **Live Interactive Demo:** [View the Live Dashboard Here](https://your-app-name.streamlit.app)

An interactive Machine Learning system built to forecast customer subscription cancellations based on customer support ticket volume, billing pricing structures, and historical account longevity.

## Tech Stack & Tools
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning Framework:** Scikit-Learn (Random Forest Classifier)
* **Application Infrastructure:** Streamlit

## Core Project Pipeline Architecture
1. **Inference Pipelines:** Loads raw business metrics data, handles missing configurations, and maps categorical text attributes to scalar input matrices.
2. **Modeling Optimization:** Implements a multi-decision tree ensemble algorithm (`RandomForestClassifier`) to predict discrete binary retention classification outputs.
3. **Serving Layer:** Exposes model prediction vectors instantly via an interactive local UI frontend canvas without external JavaScript requirements.

## Execution Instructions

### 1. Installation Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Execution of Modeling Core
```bash
python src/train.py
```

### 3. Initialize Interactive Canvas Frontend Application
```bash
streamlit run src/app.py
```
