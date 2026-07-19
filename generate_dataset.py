"""
generate_dataset.py
Creates a synthetic but realistic student performance dataset and saves it
to data/students.csv. Run once (already run — CSV is included in submission).
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 400

study_hours = np.clip(np.random.normal(5, 2.2, N), 0, 12)
attendance = np.clip(np.random.normal(75, 15, N), 30, 100)
previous_score = np.clip(np.random.normal(65, 15, N), 20, 100)
sleep_hours = np.clip(np.random.normal(6.5, 1.3, N), 3, 10)
assignments_done = np.clip(np.random.normal(7, 2.5, N), 0, 10).round()

# Underlying "true" score that determines pass/fail (with noise), so the
# model has real, learnable signal instead of pure randomness.
score = (
    2.6 * study_hours
    + 0.32 * attendance
    + 0.28 * previous_score
    + 1.1 * sleep_hours
    + 1.8 * assignments_done
    + np.random.normal(0, 9, N)
)

threshold = np.percentile(score, 42)  # roughly 58% pass rate
passed = (score > threshold).astype(int)

df = pd.DataFrame({
    "study_hours": study_hours.round(1),
    "attendance_pct": attendance.round(1),
    "previous_score": previous_score.round(1),
    "sleep_hours": sleep_hours.round(1),
    "assignments_done": assignments_done.astype(int),
    "passed": passed
})

df.to_csv("data/students.csv", index=False)
print("Saved data/students.csv with shape", df.shape)
print(df["passed"].value_counts())
