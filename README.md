# 🎓 Student Pass/Fail Predictor — AI Lab Project

A Python + Streamlit application that predicts whether a student will
**pass or fail** based on study habits (study hours, attendance, previous
score, sleep, assignments completed). It uses a **Machine Learning AI**
approach and includes a fully visual, interactive, and explainable UI, as
required by the AI Lab Project Guide.

## 1. Problem

Given a student's study habits, predict PASS/FAIL and explain *why*, so a
student or advisor can see which factors matter most and compare the
performance of two different ML models.

## 2. Setup

```bash
# 1. Clone / unzip this folder, then cd into it
cd StudentPassPredictor_AILab

# 2. (Recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```
### Installation

1. Clone the repository
```bash
git clone https://github.com/sarwerk/Student-Pass-Fail-Performance-Prediction.git
```

2. Go to the project folder
```bash
cd Student-Pass-Fail-Performance-Prediction
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
streamlit run app.py
```

## 3. Run

```bash
streamlit run app.py
```

This opens the app in your browser (default: http://localhost:8501).

The sample dataset is already included at `data/students.csv`. To
regenerate it from scratch, run:

```bash
python3 generate_dataset.py
```

## 4. How to use the app

1. **Predict a Student tab** — set sliders for study hours, attendance,
   previous score, sleep, and assignments completed. Choose a model
   (Logistic Regression or Random Forest) and click **Run Prediction**.
   The app shows the predicted outcome, confidence, and a natural-language
   explanation with a chart of the top contributing factors.
2. **Dataset Overview tab** — see the training data, class distribution,
   and a scatter plot of study hours vs attendance colored by outcome.
3. **Model Evaluation tab** — compares Logistic Regression vs Random
   Forest on accuracy, precision, recall, F1 score, and confusion
   matrices, using a held-out test split.

## 5. Project structure

```
StudentPassPredictor_AILab/
├── app.py                # Main Streamlit app (UI + logic + explainability + evaluation)
├── generate_dataset.py   # Script that created data/students.csv
├── requirements.txt
├── README.md
├── report.md              # Short project report
├── data/
│   └── students.csv      # Sample dataset (400 synthetic student records)
└── screenshots/           # Add UI screenshots here before submission
```

## 6. AI component

- **Type:** Machine Learning AI (Option 3 in the lab guide)
- **Models:** Logistic Regression and Random Forest Classifier
  (scikit-learn), trained on an 75/25 train/test split of the sample
  dataset.
- **Explainability:** Logistic Regression coefficients and Random Forest
  feature importances are used to show the top 3 factors behind each
  prediction, plus a natural-language summary.
- **Evaluation metrics:** Accuracy, Precision, Recall, F1 score, and
  Confusion Matrix, compared side-by-side for both models.

No external AI APIs are used — both models are trained locally on the
included sample dataset, so there are no external API costs, prompts, or
privacy concerns to disclose.

## 7. Known limitations & future improvements

- The dataset is synthetic; a real dataset (e.g., from an LMS) would make
  predictions more meaningful.
- Only two model types are compared; more algorithms (e.g., SVM, XGBoost)
  could be added.
- The app currently supports single-student predictions; batch CSV upload
  and prediction could be added as a future improvement.

## Features

- Predicts student pass/fail using Machine Learning.
- Interactive Streamlit web interface.
- Uses Logistic Regression and Random Forest models.
- Displays prediction results instantly.
- Easy to use and beginner-friendly.

## Requirements

- Python 3.10+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
