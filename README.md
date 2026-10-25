# Early Diabetes Risk Dataset Exploration

## Project Overview

An educational data science and machine-learning project that explores a public diabetes-related dataset, examines data quality, visualizes patterns, and compares basic classification models.

**Disclaimer:** This project is for educational purposes only. It is not a medical diagnostic or screening tool and must not be used for healthcare decisions.

## Objectives

- Inspect dataset structure, data types, missing values, and duplicate records.
- Perform exploratory data analysis (EDA).
- Visualize feature distributions and correlations.
- Establish a dummy-classifier baseline.
- Compare logistic regression and decision tree classifiers.
- Evaluate precision, recall (sensitivity), specificity, and confusion matrices.
- Discuss false positives, false negatives, bias, and model limitations.

## Technologies

- Python
- pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn

## Dataset

The script downloads a public copy of the Pima Indians Diabetes dataset from GitHub when the local CSV is absent.

Source used by the script:
https://github.com/Yuridna/pima-indians-diabetes-database

Review and cite the original dataset documentation and applicable terms before redistribution. The dataset represents a specific population, so its findings may not generalize to other populations.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn
```

Run the project:

```bash
python early_diabetes_risk_exploration.py
```

An internet connection is required to download the dataset if it is not already present locally.

## Outputs

The script generates dataset notes, model evaluation summaries, CSV metrics, and plots under the `reports/` directory.

## Limitations and Ethics

- Results depend on the dataset's quality, labels, and population.
- Treating selected zero measurements as missing is an analytical assumption that requires documentation.
- Associations do not establish causation.
- A single train/test split does not establish clinical validity.
- No real personal health data should be collected through a public demo.
- Model predictions must not be used for diagnosis, screening, or treatment decisions.

## Project Status

Educational exploration and baseline model comparison.
