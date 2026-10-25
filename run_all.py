import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score
from sklearn.calibration import calibration_curve

# 0. Download Dataset if not present
try:
    df = pd.read_csv("diabetes.csv")
except FileNotFoundError:
    print("Downloading diabetes.csv...")
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    df = pd.read_csv(url, names=columns)
    df.to_csv("diabetes.csv", index=False)

print("=" * 70)
print("PROJECT SCOPE: Strictly educational dataset exploration.")
print("This model is not a medical diagnostic or screening tool.")
print("=" * 70)

# 1. Missing Values Replacement
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df_cleaned = df.copy()
df_cleaned[zero_as_missing] = df_cleaned[zero_as_missing].replace(0, np.nan)
print("\nMissing values (NaN) count:\n", df_cleaned.isnull().sum())

# 2. EDA Visualizations
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
sns.countplot(data=df_cleaned, x="Outcome", ax=axes[0], hue="Outcome", legend=False)
axes[0].set_title("Target Distribution (0 = Neg, 1 = Pos)")

sns.histplot(data=df_cleaned, x="Glucose", hue="Outcome", kde=True, ax=axes[1], palette="Set2")
axes[1].set_title("Glucose Distribution by Outcome")

sns.boxplot(data=df_cleaned,x="Outcome",y="BMI",ax=axes[2],hue="Outcome",legend=False,)
axes[2].set_title("BMI Distribution by Outcome")
plt.tight_layout()
plt.show()

# 3. Train-Test Split
X = df_cleaned.drop(columns=["Outcome"])
y = df_cleaned["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Dummy Baseline
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)

# 5. Logistic Regression Pipeline
pipe_lr = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
pipe_lr.fit(X_train, y_train)
lr_pred = pipe_lr.predict(X_test)
lr_probs = pipe_lr.predict_proba(X_test)[:, 1]

# 6. Evaluation
cm = confusion_matrix(y_test, lr_pred)
tn, fp, fn, tp = cm.ravel()
print("\n--- Model Evaluation ---")
print(f"Dummy Accuracy:       {dummy.score(X_test, y_test):.3f}")
print(f"Sensitivity (Recall): {tp / (tp + fn):.3f}")
print(f"Specificity:          {tn / (tn + fp):.3f}")
print(f"Precision:            {precision_score(y_test, lr_pred):.3f}")
print("\nClassification Report:\n", classification_report(y_test, lr_pred))

# Confusion Matrix Plot
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred Neg", "Pred Pos"],
            yticklabels=["Actual Neg", "Actual Pos"])
plt.title("Logistic Regression Confusion Matrix")
plt.ylabel("Ground Truth")
plt.xlabel("Prediction")
plt.show()

# 7. Calibration Curve
prob_true, prob_pred = calibration_curve(y_test, lr_probs, n_bins=5)
plt.figure(figsize=(6, 4))
plt.plot(prob_pred, prob_true, marker="o", label="Logistic Regression")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Observed Fraction of Positives")
plt.title("Calibration Curve")
plt.legend()
plt.show()

# 8. Synthetic Inference
synthetic_sample = pd.DataFrame([{
    "Pregnancies": 2, "Glucose": 138.0, "BloodPressure": 72.0, "SkinThickness": 35.0,
    "Insulin": 120.0, "BMI": 32.4, "DiabetesPedigreeFunction": 0.55, "Age": 33
}])
predicted_risk = pipe_lr.predict_proba(synthetic_sample)[0, 1]
print("\n--- Synthetic Sample Prediction ---")
print(f"Calculated Statistical Risk: {predicted_risk * 100:.2f}% (Educational model only)")