# Student Dropout & Academic Success – Multi-class Classification

ML Assignment 2 (M.Tech AIML, BITS WILP) – five classifiers, evaluated and served through a Streamlit app.

## a. Problem statement

Predict, at the end of a student's second semester, whether the student will **Dropout**, remain **Enrolled**, or **Graduate** on time, using demographic, socio-economic, macro-economic and first/second-semester academic-performance features. Early identification of at-risk students lets institutions intervene sooner. This is a **3-class** classification problem; five models (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, SVM) are trained on the same data and compared with Accuracy, AUC, Precision, Recall, F1 and MCC.

## b. Dataset description

| Item | Value |
|---|---|
| Source | UCI ML Repository – *Predict Students' Dropout and Academic Success* (Realinho et al., 2021) |
| Instances | 4,424 |
| Features | 36 (all numeric / integer-coded categoricals; no missing values) |
| Target | `Target` – Dropout (1,421), Enrolled (794), Graduate (2,209) |
| Feature groups | Demographics (marital status, age, gender, nationality), application info (mode, order, course, previous qualification/grade, admission grade), family background (parents' qualification/occupation), financial flags (debtor, tuition up to date, scholarship), 1st & 2nd semester curricular units (credited, enrolled, evaluations, approved, grade, without evaluations), macro-economic (unemployment, inflation, GDP) |
| Split | 80 % train / 20 % test, stratified, `random_state=42` → 3,539 train / 885 test |

Preprocessing: whitespace stripped from column names; `StandardScaler` inside the pipeline for Logistic Regression, KNN, Naive Bayes and SVM (the Decision Tree is scale-invariant). The scaler is saved together with each model, so the app needs no separate preprocessing step.

Metric conventions (multi-class): Precision, Recall and F1 are **macro-averaged**; AUC is **one-vs-rest, macro-averaged** using predicted probabilities; MCC is the multi-class generalisation in scikit-learn.

## c. GitHub repository link

https://github.com/SumathSP/BITS_ML_Assignment_2.git

Live Streamlit app: `<PASTE STREAMLIT APP URL HERE>`

```
BITS_ML_Assignment_2/
├── app.py                  # Streamlit app
├── common.py               # shared metrics / label helpers
├── train_models.py         # trains + evaluates + saves all 5 models
├── requirements.txt
├── README.md
├── test_data.csv           # held-out 20 % split (885 rows, with Target) – upload this in the app
├── data/students_dropout_full.csv   # original dataset
└── model/                  # saved pipelines (.joblib) + metrics_summary.{csv,json}
```

Reproduce: `pip install -r requirements.txt && python train_models.py && streamlit run app.py`

## d. Models used

Hyper-parameters: Logistic Regression (`max_iter=2000`), Decision Tree (`max_depth=6, min_samples_leaf=10`), KNN (`k=11`, distance-weighted), Gaussian Naive Bayes (defaults), SVM (RBF, `C=1`, `probability=True`).

### Comparison table (test set, 885 samples)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7684 | 0.8778 | 0.7070 | 0.6754 | 0.6826 | 0.6150 |
| Decision Tree | 0.7458 | 0.8511 | 0.6890 | 0.6609 | 0.6698 | 0.5780 |
| KNN | 0.6949 | 0.8082 | 0.6245 | 0.5787 | 0.5822 | 0.4874 |
| Naive Bayes | 0.6588 | 0.7893 | 0.5675 | 0.5579 | 0.5576 | 0.4279 |
| Support Vector Machine | 0.7582 | 0.8664 | 0.7051 | 0.6676 | 0.6785 | 0.5986 |

### Observations

Per-class recall on the test set (Dropout / Enrolled / Graduate): LR 0.77/0.33/0.93, DT 0.69/0.38/0.91, KNN 0.64/0.18/0.92, NB 0.67/0.18/0.83, SVM 0.73/0.35/0.93. The minority, ambiguous *Enrolled* class is hard for every model and drags macro scores down.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best on every metric (accuracy 76.8 %, AUC 0.878, MCC 0.615). The semester-wise "approved units / grades" features are close to linearly related to the outcome, so a simple linear boundary generalises well; also the best Dropout recall (0.77). Still recalls only 33 % of *Enrolled*. |
| Decision Tree | Second-best on interpretability-vs-accuracy: 74.6 % accuracy, AUC 0.851. Depth limit (6) and leaf size (10) avoid overfitting. Has the highest *Enrolled* recall (0.38) because splits can isolate mid-performers, but its hard splits give less well-calibrated probabilities, hence lower AUC than LR/SVM. |
| KNN | Weakest of the distance-based models: accuracy 69.5 %, *Enrolled* recall only 0.18. 36 features (many sparse integer-coded categoricals) dilute Euclidean distance, and the class imbalance lets the Graduate majority dominate neighbourhoods. |
| Naive Bayes | Lowest scores (accuracy 65.9 %, MCC 0.428). Its feature-independence assumption is violated heavily – 1st and 2nd semester units/grades are strongly correlated – and Gaussian likelihoods fit the categorical-coded columns poorly. Fast, but not competitive here. |
| Support Vector Machine | Very close to LR (accuracy 75.8 %, AUC 0.866, F1 0.679, MCC 0.599). RBF kernel adds no gain over the linear model on this mostly linearly-separable problem and costs much more training/inference time. |
| **Overall winner for this dataset** | **Logistic Regression** – highest Accuracy, AUC, Precision, Recall, F1 and MCC, while being the simplest, fastest and most interpretable. SVM is a close runner-up; Naive Bayes and KNN trail. |

## Streamlit app features

1. CSV upload of test data (or one-click bundled `test_data.csv`) with column/label validation.
2. Model selection dropdown (5 models).
3. Accuracy, AUC, Precision, Recall, F1, MCC for the selected model.
4. Confusion matrix + per-class classification report.
5. Side-by-side comparison table of all five models on the uploaded file, with the best by MCC.

Models are loaded from `model/*.joblib`; nothing is retrained in the app.
