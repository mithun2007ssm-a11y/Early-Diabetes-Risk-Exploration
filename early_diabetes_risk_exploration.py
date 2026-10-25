"""
Early Diabetes Risk Dataset Exploration
Educational use only — NOT a diagnostic or screening tool.

Run from the VS Code terminal:
    python early_diabetes_risk_exploration.py

Required packages:
    python -m pip install pandas numpy matplotlib seaborn scikit-learn

The script downloads a public copy of the Pima Indians Diabetes dataset from
GitHub if data/diabetes.csv is not already present. Confirm source provenance
and permitted use before submitting your project.
"""

from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# -------------------- Configuration --------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
REPORTS_DIR = PROJECT_DIR / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"
DATA_PATH = DATA_DIR / "diabetes.csv"

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "Yuridna/pima-indians-diabetes-database/"
    "master/diabetes.csv"
)
TARGET = "Outcome"

DISCLAIMER = (
    "EDUCATIONAL ONLY. This project explores a public dataset and evaluates "
    "basic machine-learning models. It is not validated for clinical use and "
    "must not be used to diagnose, screen, or make treatment decisions."
)


def download_dataset():
    """Download diabetes.csv only if it is not already in the data folder."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_PATH.exists():
        print(f"Using existing dataset: {DATA_PATH}")
        return

    print("Downloading dataset from the configured public GitHub source...")
    try:
        with urlopen(DATA_URL, timeout=30) as response:
            content = response.read()
    except Exception as exc:
        raise RuntimeError(
            "\nDataset download failed. Check your internet connection. "
            "Alternatively, download a permitted copy of diabetes.csv and "
            "place it in the project's data folder."
        ) from exc

    if not content.startswith(b"Pregnancies,"):
        raise ValueError(
            "The downloaded file did not match the expected CSV format. "
            "Check the dataset URL before continuing."
        )

    DATA_PATH.write_bytes(content)
    print(f"Dataset saved to: {DATA_PATH}")


def save_text(filename, content):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / filename).write_text(content, encoding="utf-8")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("EARLY DIABETES RISK DATASET EXPLORATION")
    print(DISCLAIMER)
    print("=" * 72)

    # 1. Download and load data
    download_dataset()
    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(
            f"Expected target column '{TARGET}' was not found. "
            f"Columns available: {df.columns.tolist()}"
        )

    print("\n1. DATASET OVERVIEW")
    print("-" * 72)
    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst five rows:")
    print(df.head().to_string(index=False))
    print("\nData types and non-null counts:")
    df.info()
    print("\nMissing values per column:")
    print(df.isnull().sum().to_string())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print("\nSummary statistics:")
    print(df.describe().round(2).to_string())
    print("\nTarget label counts:")
    print(df[TARGET].value_counts(dropna=False).sort_index().to_string())
    print("\nTarget label proportions:")
    print(
        df[TARGET].value_counts(normalize=True, dropna=False)
        .sort_index().round(3).to_string()
    )

    # Save initial documentation and data-quality notes
    notes = f"""EARLY DIABETES RISK DATASET EXPLORATION — INITIAL NOTES

{DISCLAIMER}

Dataset source URL used by this script:
{DATA_URL}

Local file: data/diabetes.csv
Rows: {df.shape[0]}
Columns: {df.shape[1]}
Column names: {", ".join(map(str, df.columns))}
Target column: {TARGET}
Target labels observed:
{df[TARGET].value_counts(dropna=False).sort_index().to_string()}

Missing values by column:
{df.isnull().sum().to_string()}

Duplicate rows: {df.duplicated().sum()}

DATA QUALITY AND ETHICS CHECKLIST
- Read and cite the original dataset documentation and license/terms.
- Verify how the target labels were collected before interpreting them.
- This commonly used dataset represents a narrow population; results may not
  generalize to other people or settings.
- In this dataset, zero values in Glucose, BloodPressure, SkinThickness,
  Insulin, and BMI are often treated as implausible placeholders in analyses.
  This script treats those zeros as missing for analysis/modeling only. This is
  an assumption that must be discussed and verified against documentation.
- Associations are descriptive and do not establish causation.
- A single train/test split is an educational estimate, not clinical validation.
- False negatives and false positives both matter in health-related contexts.
- Do not collect real personal health information in a public demo.
- Do not use model outputs for diagnosis, screening, or treatment decisions.
"""
    save_text("initial_data_notes.txt", notes)

    # 2. Prepare an analysis copy. Keep raw df unchanged.
    print("\n2. DATA QUALITY PREPARATION")
    print("-" * 72)
    analysis_df = df.copy()

    possible_placeholder_cols = [
        c for c in
        ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        if c in analysis_df.columns
    ]
    for col in possible_placeholder_cols:
        zero_count = int((analysis_df[col] == 0).sum())
        analysis_df.loc[analysis_df[col] == 0, col] = np.nan
        print(f"{col}: treated {zero_count} zero value(s) as missing for analysis")

    # 3. Exploratory data analysis
    print("\n3. EXPLORATORY DATA ANALYSIS")
    print("-" * 72)
    sns.set_theme()

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=TARGET)
    plt.title("Target Label Distribution")
    plt.xlabel("Outcome label (as supplied by dataset)")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "target_distribution.png", dpi=150)
    plt.close()

    numeric_features = [
        c for c in analysis_df.select_dtypes(include=np.number).columns
        if c != TARGET
    ]
    if numeric_features:
        analysis_df[numeric_features].hist(figsize=(12, 9), bins=20)
        plt.suptitle("Feature Distributions")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "feature_distributions.png", dpi=150)
        plt.close()

    selected_box_cols = [
        c for c in ["Glucose", "BMI", "Age", "BloodPressure"]
        if c in analysis_df.columns
    ]
    if selected_box_cols:
        fig, axes = plt.subplots(
            1, len(selected_box_cols),
            figsize=(4 * len(selected_box_cols), 4)
        )
        if len(selected_box_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, selected_box_cols):
            sns.boxplot(data=analysis_df, x=TARGET, y=col, ax=ax)
            ax.set_title(f"{col} by outcome label")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "selected_features_by_outcome.png", dpi=150)
        plt.close()

    corr = analysis_df.select_dtypes(include=np.number).corr()
    plt.figure(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (association is not causation)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    print("Saved EDA plots to:", PLOTS_DIR)
    print("Review the plots and write down observations without claiming causation.")

    # 4. Train/test split
    print("\n4. BASELINE AND CLASSIFIER EVALUATION")
    print("-" * 72)
    model_df = analysis_df.dropna(subset=[TARGET]).copy()
    X = model_df.drop(columns=[TARGET])
    y = model_df[TARGET]

    if y.nunique() != 2:
        raise ValueError("This script expects a binary target with two classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    # Imputation/scaling are fitted only on training data via pipelines.
    models = {
        "Dummy baseline": DummyClassifier(strategy="prior"),
        "Logistic regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
        ]),
        "Decision tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(max_depth=4, random_state=42)),
        ]),
    }

    all_results = []
    report_parts = [
        "MODEL EVALUATION SUMMARY",
        "",
        DISCLAIMER,
        "",
        "Metrics below describe this dataset and one test split only. They are "
        "not evidence of clinical suitability.",
        "",
    ]

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, zero_division=0)
        sensitivity = recall_score(y_test, pred, zero_division=0)
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) else float("nan")

        all_results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision (label 1)": precision,
            "Sensitivity / Recall (label 1)": sensitivity,
            "Specificity (label 0)": specificity,
            "True negatives": tn,
            "False positives": fp,
            "False negatives": fn,
            "True positives": tp,
        })

        report_parts.extend([
            f"--- {name} ---",
            f"Accuracy: {accuracy:.3f}",
            f"Precision (label 1): {precision:.3f}",
            f"Sensitivity / recall (label 1): {sensitivity:.3f}",
            f"Specificity (label 0): {specificity:.3f}",
            f"Confusion matrix (rows=true labels 0,1; columns=predicted 0,1):\n{cm}",
            "Classification report:",
            classification_report(y_test, pred, zero_division=0),
            "Interpretation: false negatives and false positives both matter. "
            "This model is not validated for clinical use.",
            "",
        ])

        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["Actual 0", "Actual 1"]
        )
        plt.title(f"Confusion Matrix — {name}")
        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_")
        plt.savefig(PLOTS_DIR / f"confusion_matrix_{safe_name}.png", dpi=150)
        plt.close()

    results_df = pd.DataFrame(all_results)
    print("\nEvaluation metrics:")
    print(results_df.round(3).to_string(index=False))

    results_df.to_csv(REPORTS_DIR / "model_metrics.csv", index=False)
    report_parts.append("COMPARISON TABLE")
    report_parts.append(results_df.round(3).to_string(index=False))
    save_text("model_evaluation_summary.txt", "\n".join(report_parts))

    # 5. Final reminder and output locations
    print("\n5. OUTPUTS CREATED")
    print("-" * 72)
    print(f"Dataset: {DATA_PATH}")
    print(f"Initial notes: {REPORTS_DIR / 'initial_data_notes.txt'}")
    print(f"Model metrics CSV: {REPORTS_DIR / 'model_metrics.csv'}")
    print(f"Evaluation summary: {REPORTS_DIR / 'model_evaluation_summary.txt'}")
    print(f"Plots: {PLOTS_DIR}")
    print("\nNext: inspect every plot, verify dataset documentation, and write your")
    print("own interpretation and limitations in the final report.")
    print("\n" + DISCLAIMER)


if __name__ == "__main__":
    main()
