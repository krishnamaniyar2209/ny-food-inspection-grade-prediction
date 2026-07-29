# 🍽️ New York State Food Safety Inspection Grade Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Models-red?logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![University](https://img.shields.io/badge/Pace%20University-CS677-blue)

> Predicting food-establishment inspection grades (A / B / C) across New York State by comparing 6 classifiers, with feature engineering, GridSearchCV tuning, and a documented target-leakage investigation that cut reported accuracy from 91% to an honest 80%. Built for CS677: Machine Learning at Pace University (Fall 2025).

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Models & Results](#-models--results)
- [Data Leakage: Identified & Fixed](#-data-leakage-identified--fixed)
- [Limitations & Next Steps](#-limitations--next-steps)
- [Installation](#-installation)
- [Usage](#-usage)
- [Key Findings](#-key-findings)
- [Technologies Used](#-technologies-used)
- [Author](#-author)

---

## 🔬 Overview

This project compares **6 machine-learning classifiers** to predict whether a New York State food establishment receives Grade A, B, or C, covering EDA, feature engineering, a **leakage-free establishment-grouped split**, model comparison, SGD optimization, and GridSearchCV tuning.

- ✅ Multiclass classification (A / B / C)
- ✅ 8 visualizations (distribution, geography, time series, grade breakdowns)
- ✅ 8 engineered features (history, geography, timing, domain indicators)
- ✅ **Establishment-grouped 60/20/20 split** (no establishment appears in two sets)
- ✅ 5 models plus an SGD optimizer, with GridSearchCV tuning
- ✅ Group K-Fold cross-validation and full metric evaluation

---

## 📊 Dataset

**Source:** New York State Open Data — *Food Safety Inspections: Current Ratings* ([data.ny.gov](https://data.ny.gov/), NYS Department of Agriculture and Markets)

| Property | Details |
|---|---|
| Original Records | ~136,900 |
| Sampled Records | **41,075** (30%, `random_state=42`) |
| Raw Columns | 13 |
| Features Used | 14 |
| Target | `Inspection Grade` (A / B / C) |
| Class Balance | **C 52.66% · B 24.32% · A 23.01%** |
| Date Range | 2023 to 2025 |
| Geographic Coverage | **62 counties, all of New York State** |

> ⚠️ **Scope note.** This dataset covers food-service establishments **statewide**, not only New York City. Counties present range from Kings and Bronx to Erie (Buffalo), Suffolk, Nassau, and Westchester. The `Is_NYC` feature exists precisely to distinguish the five boroughs from the remaining 57 counties. Establishment types include groceries and convenience stores alongside restaurants, which is why the Grade C share is far higher than NYC restaurant-only figures would suggest.

### Data Quality

| Column | Missing | % |
|---|---|---|
| `Deficiency Description` | 9,452 | 23.01% |
| `Deficiency Number` | 9,452 | 23.01% |
| `Georeference` | 219 | 0.53% |
| `Trade Name` | 51 | 0.12% |
| `Establishment Types` | 1 | 0.00% |

19,175 missing values across 533,975 cells (3.6%). Coordinates and timing gaps were median-imputed; `Failure_Rate` was filled with 0.0 for first-time establishments.

### Engineered Features

| Feature | Description | RF Importance |
|---|---|---|
| `Previous_Grade_Encoded` | Prior inspection grade (backward lag) | **36.8%** ⭐ |
| `Failure_Rate` | **Prior-only** historical Grade-C rate | 24.6% |
| `Total_Inspections` | Cumulative inspection count | 11.7% |
| `Days_Since_Last` | Days since previous inspection | 11.1% |
| `Longitude` + `Latitude` | Parsed from `Georeference` POINT string | 6.3% |
| `Has_Multiple_Types` | Character length of the establishment-type code, a proxy for how many type codes apply (e.g. `A` vs `ABC`) | 1.6% |
| `Is_NYC` | Whether the county is one of the five NYC boroughs | 1.4% |

Plus `County_Encoded`, `Establishment_Encoded`, and four date parts (`Year`, `Month`, `Quarter`, `DayOfWeek`), for 14 features total.

---

## 📁 Project Structure
```
ny-food-inspection-grade-prediction/
│
├── NYC_Food_Safety_Inspection_Grade_Prediction.ipynb  # Main notebook
├── README.md                                          # Documentation
└── requirements.txt                                   # Dependencies
```

---

## 🔬 Methodology

1. **EDA** with 8 visualizations covering grade distribution, geography, establishment type, and time
2. **Feature engineering**, 8 features, with all history features (`Failure_Rate`, `Previous_Grade`, `Total_Inspections`, `Days_Since_Last`) computed **strictly from prior inspections** after sorting by date
3. **Grouped split** via `GroupShuffleSplit` keyed on `Trade Name || Street`, so no establishment appears in more than one set
   - Train 24,864 · Validation 8,069 · Test 8,142
4. **Scaling** with `StandardScaler` fit on training data only
5. **Models**: Decision Tree, Random Forest, Logistic Regression, SVM (Linear and RBF), plus an `SGDClassifier` trained with `partial_fit` over 100 epochs
6. **Validation** with `GroupKFold` (establishment-aware) cross-validation
7. **Tuning** with `GridSearchCV` on Decision Tree and Random Forest

### Baseline

**Grade C accounts for 52.66% of records.** A trivial classifier that always predicts C scores 52.66% accuracy. Every result below should be read against that floor.

---

## 📈 Models & Results

### Default Model Performance

| Model | Train Acc | Val Acc | Test Acc | Train−Test | Status |
|---|---|---|---|---|---|
| Random Forest | 0.9998 | 0.7908 | 0.7886 | 0.2112 | ⚠️ Overfitting |
| Logistic Regression | 0.7732 | 0.7681 | 0.7684 | 0.0048 | ✅ Good fit |
| Decision Tree | 0.9998 | 0.7564 | 0.7577 | 0.2422 | ⚠️ Overfitting |
| SVM RBF | 0.6109 | 0.6058 | 0.5985 | 0.0124 | ⚠️ Underfitting |
| SVM Linear | 0.2744 | 0.2854 | 0.2891 | −0.0147 | ⚠️ Underfitting / non-convergent |

Untuned tree models memorize the training set almost perfectly (0.9998) while generalizing to roughly 0.79. That 21 to 24 point gap is what motivates the tuning step below.

**SVM Linear is not a real result.** At 28.91% test accuracy it performs worse than random guessing across three classes and roughly half the majority-class baseline. See the convergence note below.

### Group K-Fold Cross-Validation (k=5)

| Model | CV Mean | CV Std |
|---|---|---|
| Random Forest | **0.7890** | 0.0036 |
| Logistic Regression | 0.7728 | 0.0029 |
| Decision Tree | 0.7617 | 0.0071 |
| SVM RBF | 0.6161 | 0.0310 |
| SVM Linear | **0.5158** | 0.0524 |

> **Both SVMs were capped at `max_iter=1000` against 24,864 training samples and did not converge.** SVM Linear at 0.5158 CV performs worse than the 0.5266 majority-class baseline — and its test accuracy is 0.2891, a **23-point gap from its own cross-validation score**. That spread is itself the clearest evidence of non-convergence: the model is not settling on a stable decision boundary between folds. Raising `max_iter` or switching to `LinearSVC` would give both SVMs a fair hearing. As run, neither is a meaningful contender, and their numbers are reported only for completeness.

### Final Performance (After Tuning)

| Rank | Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 1 | **Random Forest (Tuned)** | **0.7973** | 0.8178 | 0.7973 | 0.8013 |
| 2 | **Decision Tree (Tuned)** | 0.7951 | **0.8263** | 0.7951 | **0.8017** |
| 3 | Logistic Regression | 0.7684 | 0.7800 | 0.7684 | 0.7711 |
| 4 | SGD | 0.7648 | 0.7781 | 0.7648 | 0.7683 |
| 5 | SVM RBF | 0.5985 | 0.6660 | 0.5985 | 0.5846 |
| 6 | SVM Linear | 0.2891 | 0.3163 | 0.2891 | 0.2973 |

Precision, recall, and F1 are weighted averages. Ranks 5 and 6 are the non-converged SVMs, included for completeness rather than as candidates.

**Best parameters**

```python
DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, min_samples_split=50)
RandomForestClassifier(max_depth=10, n_estimators=100)
```

**Which model actually wins is genuinely ambiguous.** Random Forest leads on accuracy by 0.0022, a margin well inside cross-validation noise. Decision Tree leads on both precision (0.8263 vs 0.8178) and F1 (0.8017 vs 0.8013), and it is a **depth-5 tree**, meaning every prediction can be traced through at most five human-readable splits. For a regulatory application where an inspector must justify why an establishment was flagged, that interpretability is worth more than 0.2 points of accuracy. **The Decision Tree is the better deployment candidate.**

Tuning also fixed the overfitting: the train-test gap fell from 0.24 to 0.0071 (Decision Tree) and 0.0375 (Random Forest).

### Per-Class Performance

**Random Forest (Tuned)**

| Grade | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| A | 0.61 | 0.74 | 0.67 | 1,923 |
| B | **0.97** | 0.72 | 0.83 | 2,118 |
| C | 0.84 | 0.86 | 0.85 | 4,101 |

**Decision Tree (Tuned)**

| Grade | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| A | 0.58 | 0.78 | 0.67 | 1,923 |
| B | **0.99** | 0.72 | 0.83 | 2,118 |
| C | 0.85 | 0.84 | 0.85 | 4,101 |

**Grade B is the most interesting row.** Precision of 0.97 to 0.99 against recall of 0.72 means that when either model predicts B it is almost always right, but it misses more than a quarter of actual Grade-B establishments. The models are conservative about assigning B and push borderline cases into A or C. Grade A is the weakest overall, with precision near 0.60: the models over-predict A, flagging establishments as clean that turn out not to be. For a food-safety use case, that is the more costly error direction and would warrant class weighting or threshold adjustment.

### Feature Importance (Tuned Random Forest)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `Previous_Grade_Encoded` | 36.78% |
| 2 | `Failure_Rate` (prior-only) | 24.57% |
| 3 | `Total_Inspections` | 11.71% |
| 4 | `Days_Since_Last` | 11.08% |
| 5 | `Longitude` | 3.44% |
| 6 | `Latitude` | 2.85% |
| 7 | `Establishment_Encoded` | 2.03% |
| 8 | `County_Encoded` | 1.71% |
| 9 | `Has_Multiple_Types` | 1.63% |
| 10 | `Is_NYC` | 1.40% |

**Inspection history dominates.** The four history features account for 84.1% of total importance. Location contributes 9.4% once `Is_NYC` is counted alongside coordinates and county, and establishment characteristics under 4%. Where an establishment is and what type it is matter far less than how it has performed before.

---

## 🛡️ Data Leakage: Identified & Fixed

An earlier version of this project reported **~91% accuracy**. That number was inflated by **target leakage**. The fix is documented here because finding and removing leakage is part of the work.

**The leak, in two parts:**

1. `Failure_Rate` divided each establishment's *total* Grade-C count by its *total* inspection count, **including the inspection being predicted**. The feature therefore contained the answer.
2. The data was split **randomly by row**, so the same establishment could appear in both train and test. The model could effectively memorize establishments rather than learn generalizable patterns.

**The fix:**

1. **Prior-only history.** `Failure_Rate` is now computed from an establishment's *earlier* inspections only, using a cumulative count shifted to exclude the current row. The same rule applies to `Previous_Grade`, `Total_Inspections`, and `Days_Since_Last`, all computed after sorting by inspection date.
2. **Grouped split.** `GroupShuffleSplit` and `GroupKFold` keyed on `Trade Name || Street`, so no establishment spans train and test. Verified: zero overlap.

**Result:** honest accuracy settled at **~80%**, down from a leaky 91%. The top predictor shifted from the leaky `Failure_Rate` to the legitimate `Previous_Grade`. **The lower number is the trustworthy one.**

Against the 52.66% majority-class baseline, 79.73% represents a genuine **+27 point** improvement.

> **The leaky version is not committed.** The notebook in this repository contains only the corrected pipeline, so the 91% figure cannot be reproduced from what is here. Adding an appendix cell that deliberately recreates the leaky `Failure_Rate` and the random row split, then reports the inflated score side by side, is on the list below.

> **One leak remains.** `GridSearchCV` and `learning_curve` both use plain `cv=5` rather than grouped folds, so establishments do leak across folds during hyperparameter selection. The reported test accuracy is unaffected, because the test set is properly grouped and held out. But the tuning CV scores (0.7990 and 0.7945) are mildly optimistic. The fix is one line, listed in Next Steps below.

---

## ⚠️ Limitations & Next Steps

1. **Grouped CV in the tuning step.** Replace `cv=5` in both `GridSearchCV` calls with `cv=list(GroupKFold(n_splits=5).split(X_train_scaled, y_train, groups_train))` to make the pipeline fully leakage-free.
2. **Demonstrate the leak rather than assert it.** Add an appendix cell computing the leaky `Failure_Rate` and a random row split so the 91% → 80% correction is reproducible from this repository.
3. **SVMs never converged.** Both were capped at `max_iter=1000` on 24,864 samples. SVM Linear's 23-point CV-to-test gap confirms it. Raise the cap or switch to `LinearSVC` to give them a fair comparison.
4. **Grade A over-prediction is the costly error.** Precision of 0.58 to 0.61 on Grade A means many establishments are predicted clean when they are not. Class weighting or per-class threshold tuning would trade some accuracy for the safer error direction.
5. **Only 30% of the data was used.** Sampling was for computational convenience. The full ~137,000 records would tighten estimates and may improve the minority-class performance.
6. **`Deficiency Description` is unused.** 77% of rows carry a free-text violation description, which is likely the richest untapped signal in the dataset. TF-IDF or embeddings over that field is the highest-value next feature.
7. **Establishment grouping is approximate.** The key `Trade Name || Street` will split a chain across locations correctly, but may fail on inconsistent name spellings, which would let a small number of establishments cross the train-test boundary.
8. **No temporal validation.** Data spans 2023 to 2025 and the split is random by establishment rather than by time. Training on 2023 to 2024 and testing on 2025 would better simulate real deployment.

---

## ⚙️ Installation
```bash
git clone https://github.com/krishnamaniyar2209/ny-food-inspection-grade-prediction.git
cd ny-food-inspection-grade-prediction
pip install -r requirements.txt
jupyter notebook NYC_Food_Safety_Inspection_Grade_Prediction.ipynb
```

---

## 🚀 Usage
1. Download the dataset from [NY State Open Data](https://data.ny.gov/) (*Food Safety Inspections: Current Ratings*) as a `.csv`
2. Open the notebook in Jupyter or Google Colab and update the path in the `pd.read_csv(...)` call — the third cell, immediately below the imports. It is currently hardcoded to a Colab path (`/content/...`)
3. Run all cells top to bottom. EDA, models, tuning, and the leakage check generate automatically

---

## 💡 Key Findings

- **Inspection history is 84% of the signal.** `Previous_Grade` (36.8%), `Failure_Rate` (24.6%), `Total_Inspections` (11.7%), and `Days_Since_Last` (11.1%) dominate. Past performance predicts future performance far better than location or establishment type
- **Removing target leakage cut accuracy from 91% to 80%**, which is the realistic, deployable benchmark. Against a 52.66% majority-class baseline, that is still a +27 point gain
- **A depth-5 Decision Tree matches a Random Forest.** 0.7951 vs 0.7973 accuracy, with higher precision and F1, at a fraction of the complexity and with full interpretability
- **Grade B predictions are near-certain but incomplete.** Precision 0.97 to 0.99, recall 0.72. Grade A is the weak point, with precision near 0.60
- **Both SVMs failed to converge** at `max_iter=1000`. SVM Linear scored 0.2891 on test against 0.5158 in cross-validation, a gap that is itself the diagnostic
- **Geography shows strong patterns.** Kings (64.1%) and Bronx (64.4%) carry the highest Grade-C rates; Suffolk (34.8%) and Erie (40.7%) the lowest
- **Grades are drifting slowly better.** Grade C fell from 54.3% (2023) to 51.7% (2025), while Grade B rose from 22.2% to 25.5%
- **Establishment type matters.** Type `A` establishments are 55.7% Grade A, while type `AC` are only 18.9%

---

## 🛠️ Technologies Used
| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| scikit-learn | Models, metrics, GridSearchCV, grouped CV |
| pandas / NumPy | Data manipulation and feature engineering |
| Matplotlib / Seaborn | Visualization |
| Jupyter Notebook | Development environment |

---

## 👤 Author

**Krishna Maniyar**, Data Analyst
- 🎓 Pace University, Seidenberg School of CSIS, MS in Data Science
- 📘 CS677: Machine Learning (Fall 2025)
- 📧 maniyarkrishnakm22@gmail.com
- 🔗 [GitHub](https://github.com/krishnamaniyar2209) · [LinkedIn](https://www.linkedin.com/in/krishnamaniyar/) · [Portfolio](https://krishnamaniyar2209.github.io/)

---

<p align="center">Made with ❤️ for CS677 @ Pace University</p>
