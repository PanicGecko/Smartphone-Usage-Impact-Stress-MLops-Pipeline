# Social Media Stress Prediction - MLOps Pipeline

## Project Overview
This project builds an end-to-end, production-ready Machine Learning Operations (MLOps) pipeline. It takes a raw dataset and fully incorporates modern ML infrastructure including version control, experiment tracking, automated testing, continuous integration/continuous deployment (CI/CD), and data drift monitoring. 

## Dataset Description
**Dataset used:** Smartphone Usage and Addiction Analysis Dataset.
- **Source:** Simulated/Kaggle dataset tailored for analyzing user behavior.
- **Prediction Task:** A classification task predicting the user's `stress_level` ("Low", "Medium", "High") based on various numerical and categorical metrics tracking their digital behaviors. 
- **Features:** Includes a mix of numeric fields (`age`, `daily_screen_time_hours`, `social_media_hours`, `gaming_hours`, `work_study_hours`, `sleep_hours`, `notifications_per_day`, `app_opens_per_day`, `weekend_screen_time`) and categorical columns (`gender`, `academic_work_impact`).

## Repository Structure
- `src/`: Python source code, separated into logic modules (`data_preprocessing.py`, `train.py`, `monitor_drift.py`, etc.).
- `data/`: Directory where the DVC-controlled dataset and drift data is mounted. 
- `tests/`: End-to-end Pytest suite containing unit, data validation, and model validation tests.
- `.github/workflows/`: Contains `ml-pipeline.yml` for automated CI/CD runs upon push/PRs.
- `reports/`: HTML and JSON artifacts written out when identifying feature drift in simulated data sets.

## Setup Instructions

### 1. Clone & Setup Environment
```bash
# Clone the repository
git clone <your-repo-link>
cd Social_Media_Stress_Prediction_MLops_Pipeline

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # (On Windows use `.venv\Scripts\activate`)

# Install dependencies
pip install -r requirements.txt
```

### 2. Pull the Dataset via DVC
The dataset is tracked via Data Version Control (DVC) and hosted on a public S3 bucket.
```bash
dvc pull
```

## Running the Pipeline

### How to Run Tests
The automated test suite runs unit tests on preprocessing, validation tests on the incoming data format, and model validation tests.
```bash
pytest tests/ -v
```

### How to Run Training
*(Note: Be sure your config YAML/JSON maps to the right variables depending on your setup!)*
```bash
python src/train.py
```

### How to Run Drift Monitoring
Evidently is used to measure drift against a reference dataset and a current dataset (representing incoming simulated "production" data).
```bash
python src/monitor_drift.py data/main/Smartphone_Usage_And_Addiction_Analysis_3600_Rows.csv data/drift/month1_data.csv
```

---

## Data Drift Analysis & Monitoring Strategy
We compared our reference dataset against the month-1 dataset to detect data shifts before they caused silent model degradation.

**1. Which features showed drift and why?**
- Features drifted: `sleep_hours`, `social_media_hours`, `transaction_id`, and `user_id` (4 out of 16 features drifted).
- *Why:* The synthetic month-1 dataset simulates a shift in population behavior—users are sleeping significantly less and spending more time on social media. Furthermore, tracking features like `transaction_id` and `user_id` showed drift because these are sequential identifiers that inherently shift over time as new users are recorded.

**2. Would this drift likely affect model performance?**
- **Yes.** While the identifier shifts (`user_id`, `transaction_id`) shouldn't inherently affect model predictions (as they should be excluded from training), shifts in core behavioral features like `sleep_hours` and `social_media_hours` are likely strong indicators of the target variable (`stress_level`). If the population distribution has fundamentally changed across these axes, the model's predictive accuracy will likely degrade since the baseline assumptions it learned have fundamentally shifted.

**3. What action would you recommend?**
- **Immediate action:** Exclude `user_id` and `transaction_id` from drift-prediction features since they are merely sequential metadata. 
- **Long-term action:** The drift in real user behavior fields (`sleep_hours`, `social_media_hours`) breached the 20% warning threshold (25.0% drifted overall). I strongly recommend **retraining** the model using the newer `month1_data.csv` so it can learn the updated behavioral representations and accurately reflect new stress levels.
