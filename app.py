"""
app.py
AI Lab Project — Student Pass/Fail Predictor
---------------------------------------------
A Streamlit application that predicts whether a student will PASS or FAIL
based on study habits, using a Machine-Learning AI approach (Option 3 from
the lab guide). It includes a visual UI, an explainability panel, and an
evaluation module comparing two ML models (Logistic Regression vs Random
Forest).

Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

FEATURES = ["study_hours", "attendance_pct", "previous_score",
            "sleep_hours", "assignments_done"]
TARGET = "passed"

# --------------------------------------------------------------------------
# A) PROBLEM SETUP MODULE
# --------------------------------------------------------------------------

def load_data(path="data/students.csv"):
    """Load the sample dataset used to train the models."""
    return pd.read_csv(path)


def validate_inputs(study_hours, attendance, previous_score,
                     sleep_hours, assignments_done):
    """Validate user-entered feature values before running the model."""
    errors = []
    if not (0 <= study_hours <= 16):
        errors.append("Study hours must be between 0 and 16.")
    if not (0 <= attendance <= 100):
        errors.append("Attendance % must be between 0 and 100.")
    if not (0 <= previous_score <= 100):
        errors.append("Previous score must be between 0 and 100.")
    if not (0 <= sleep_hours <= 14):
        errors.append("Sleep hours must be between 0 and 14.")
    if not (0 <= assignments_done <= 10):
        errors.append("Assignments done must be between 0 and 10.")
    return errors


# --------------------------------------------------------------------------
# B) CORE LOGIC MODULE
# --------------------------------------------------------------------------

def preprocess_data(df):
    """Split into features/target and scale features for Logistic Regression."""
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled,
        "scaler": scaler,
    }


@st.cache_resource(show_spinner=False)
def run_model_or_algorithm(_df_hash):
    """
    Train both models (Logistic Regression + Random Forest) once and cache
    them along with their test-set metrics. Intermediate steps (train/test
    split, scaling) are exposed via the returned dict so the UI can show
    them, not just the final prediction.
    """
    df = load_data()
    prep = preprocess_data(df)

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(prep["X_train_scaled"], prep["y_train"])

    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    rf.fit(prep["X_train"], prep["y_train"])

    models = {"Logistic Regression": log_reg, "Random Forest": rf}
    metrics = {}
    preds = {}
    for name, model in models.items():
        X_te = prep["X_test_scaled"] if name == "Logistic Regression" else prep["X_test"]
        y_pred = model.predict(X_te)
        preds[name] = y_pred
        metrics[name] = {
            "accuracy": accuracy_score(prep["y_test"], y_pred),
            "precision": precision_score(prep["y_test"], y_pred),
            "recall": recall_score(prep["y_test"], y_pred),
            "f1": f1_score(prep["y_test"], y_pred),
            "confusion_matrix": confusion_matrix(prep["y_test"], y_pred),
        }

    return {
        "df": df, "prep": prep, "models": models,
        "metrics": metrics, "preds": preds,
    }


def predict_student(bundle, model_name, student_input):
    """Run a single prediction for a user-entered student profile."""
    scaler = bundle["prep"]["scaler"]
    model = bundle["models"][model_name]
    X_input = pd.DataFrame([student_input], columns=FEATURES)
    if model_name == "Logistic Regression":
        X_use = scaler.transform(X_input)
    else:
        X_use = X_input
    pred = model.predict(X_use)[0]
    proba = model.predict_proba(X_use)[0]
    return pred, proba

Commit 2
Prediction button ke upar comment.
# Predict student result
Commit
refactor: improve code readability

# --------------------------------------------------------------------------
# D) EXPLAINABILITY MODULE
# --------------------------------------------------------------------------

def generate_explanation(model_name, model, student_input, pred, proba):
    """
    Produce a short natural-language explanation plus the key factors that
    drove the prediction, so the user can see WHY the app decided this.
    """
    confidence = max(proba) * 100
    verdict = "PASS" if pred == 1 else "FAIL"

    if model_name == "Logistic Regression":
        coefs = model.coef_[0]
        importances = dict(zip(FEATURES, coefs))
    else:
        importances = dict(zip(FEATURES, model.feature_importances_))

    ranked = sorted(importances.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top_factors = ranked[:3]

    friendly_names = {
        "study_hours": "study hours",
        "attendance_pct": "attendance",
        "previous_score": "previous exam score",
        "sleep_hours": "sleep hours",
        "assignments_done": "assignments completed",
    }

    reasons = []
    for feat, weight in top_factors:
        direction = "raises" if weight > 0 else "lowers"
        reasons.append(f"{friendly_names[feat]} strongly {direction} the pass likelihood")

    text = (
        f"The model predicts **{verdict}** with {confidence:.1f}% confidence. "
        f"The most influential factors were: {', '.join(reasons)}."
    )
    return text, top_factors, friendly_names


# --------------------------------------------------------------------------
# C) VISUAL UI MODULE — chart helpers
# --------------------------------------------------------------------------

def create_visuals_dataset_overview(df):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    df["passed"].map({0: "Fail", 1: "Pass"}).value_counts().plot(
        kind="bar", ax=ax, color=["#e74c3c", "#2ecc71"]
    )
    ax.set_title("Pass / Fail distribution in training data")
    ax.set_xlabel("")
    ax.set_ylabel("Number of students")
    plt.xticks(rotation=0)
    plt.tight_layout()
    return fig


def create_visuals_scatter(df):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    colors = df["passed"].map({0: "#e74c3c", 1: "#2ecc71"})
    ax.scatter(df["study_hours"], df["attendance_pct"], c=colors, alpha=0.6, s=25)
    ax.set_xlabel("Study hours / week")
    ax.set_ylabel("Attendance %")
    ax.set_title("Study hours vs Attendance (green = passed)")
    plt.tight_layout()
    return fig


def create_visuals_feature_importance(top_factors, friendly_names):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    names = [friendly_names[f] for f, _ in top_factors][::-1]
    vals = [w for _, w in top_factors][::-1]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in vals]
    ax.barh(names, vals, color=colors)
    ax.set_title("Top factors behind this prediction")
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    return fig


def create_visuals_confusion_matrix(cm, model_name):
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="black", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Fail", "Pass"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Fail", "Pass"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    return fig


def create_visuals_metric_comparison(metrics):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    model_names = list(metrics.keys())
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    width = 0.2
    x = np.arange(len(metric_keys))
    for i, name in enumerate(model_names):
        vals = [metrics[name][k] for k in metric_keys]
        ax.bar(x + i * width, vals, width, label=name)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison on held-out test data")
    ax.legend()
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------
# E) EVALUATION MODULE is rendered inside render_ui() using the metrics
#    computed in run_model_or_algorithm()
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# RENDER UI
# --------------------------------------------------------------------------

def render_ui():
    st.set_page_config(page_title="Student Pass/Fail Predictor", layout="wide")
    st.title("🎓 Student Pass/Fail Predictor")
    st.caption(
        "AI Lab Project — Machine Learning AI (classification) with a visual, "
        "explainable interface. Compares Logistic Regression vs Random Forest."
    )

    bundle = run_model_or_algorithm(_df_hash="v1")
    df = bundle["df"]

    tab1, tab2, tab3 = st.tabs(
        ["🧮 Predict a Student", "📊 Dataset Overview", "📈 Model Evaluation"]
    )

    # ---------------- TAB 1: Problem Setup + Prediction ----------------
    with tab1:
        st.subheader("1. Enter student data")
        col1, col2 = st.columns([1, 1.4])

        with col1:
            model_name = st.selectbox(
                "Choose model", list(bundle["models"].keys())
            )
            study_hours = st.slider("Study hours / week", 0.0, 16.0, 5.0, 0.5)
            attendance = st.slider("Attendance %", 0.0, 100.0, 75.0, 1.0)
            previous_score = st.slider("Previous exam score", 0.0, 100.0, 65.0, 1.0)
            sleep_hours = st.slider("Sleep hours / night", 0.0, 14.0, 6.5, 0.5)
            assignments_done = st.slider("Assignments completed (out of 10)", 0, 10, 7)

            run = st.button("▶ Run Prediction", type="primary")

        with col2:
            if run:
                errors = validate_inputs(
                    study_hours, attendance, previous_score,
                    sleep_hours, assignments_done
                )
                if errors:
                    for e in errors:
                        st.error(e)
                else:
                    with st.spinner("Running model..."):
                        student_input = {
                            "study_hours": study_hours,
                            "attendance_pct": attendance,
                            "previous_score": previous_score,
                            "sleep_hours": sleep_hours,
                            "assignments_done": assignments_done,
                        }
                        pred, proba = predict_student(bundle, model_name, student_input)
                        model = bundle["models"][model_name]
                        explanation, top_factors, friendly_names = generate_explanation(
                            model_name, model, student_input, pred, proba
                        )

                    st.success("Prediction complete ✅")

                    result_col1, result_col2 = st.columns([1, 1])
                    with result_col1:
                        verdict = "PASS 🎉" if pred == 1 else "FAIL ⚠️"
                        st.metric("Predicted outcome", verdict,
                                  f"{max(proba)*100:.1f}% confidence")
                        st.progress(float(proba[1]), text=f"P(Pass) = {proba[1]*100:.1f}%")

                    with result_col2:
                        st.markdown("**Explanation**")
                        st.info(explanation)

                    st.markdown("**Why the model decided this:**")
                    fig = create_visuals_feature_importance(top_factors, friendly_names)
                    st.pyplot(fig)
            else:
                st.markdown(
                    "👈 Set the student's profile and click **Run Prediction** "
                    "to see the visual result, explanation, and confidence."
                )

    # ---------------- TAB 2: Dataset / Problem context ----------------
    with tab2:
        st.subheader("Training dataset (sample)")
        st.dataframe(df.head(15), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(create_visuals_dataset_overview(df))
        with c2:
            st.pyplot(create_visuals_scatter(df))

        st.caption(
            f"Dataset: {len(df)} synthetic student records with 5 features "
            f"(study hours, attendance, previous score, sleep, assignments) "
            f"and a binary pass/fail label."
        )

    # ---------------- TAB 3: Evaluation Module ----------------
    with tab3:
        st.subheader("Model comparison (Logistic Regression vs Random Forest)")
        st.pyplot(create_visuals_metric_comparison(bundle["metrics"]))

        m1, m2 = st.columns(2)
        for col, name in zip([m1, m2], bundle["models"].keys()):
            with col:
                st.markdown(f"**{name}**")
                mets = bundle["metrics"][name]
                st.write(f"Accuracy: {mets['accuracy']:.3f}")
                st.write(f"Precision: {mets['precision']:.3f}")
                st.write(f"Recall: {mets['recall']:.3f}")
                st.write(f"F1 score: {mets['f1']:.3f}")
                st.pyplot(create_visuals_confusion_matrix(mets["confusion_matrix"], name))


if __name__ == "__main__":
    render_ui()
