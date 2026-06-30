# summer26-diabetes-prediction
Team project: summer26-diabetes-prediction

## Checkpoint 1: Problem Definition, Data Gathering, and KPIs
**Team Members:** Xiangyi Tao, Yanli Liu, Bilal Aytekin


## 1. Problem Definition

**Question:** Can we predict whether an individual has diabetes using self-reported health, lifestyle, and demographic survey data?

**Decision/Action Informed:**
This project supports early diabetes risk screening. The model is not intended to diagnose diabetes clinically, but it can help identify people who may benefit from follow-up testing, lifestyle counseling, or preventive care. Because the features are based on survey-style health indicators, the workflow is suitable for population-level screening and public health prioritization.

**Stakeholders:**
*   **Public Health Officials (e.g., CDC, State Health Departments):** Care about population health trends, resource allocation, and identifying important risk factors for prevention campaigns.
*   **Healthcare Providers (Doctors, Clinics):** Care about efficiently identifying patients who may need additional screening, such as A1C or fasting blood glucose tests.
*   **Health Insurance Companies:** Care about reducing long-term costs from diabetes-related complications through earlier intervention.
*   **Patients/Individuals:** Care about earlier awareness of diabetes risk so they can seek care and make lifestyle changes.

**Unit of Analysis:**
An individual survey respondent.

**Scope and Boundaries:**
*   **Population:** United States adults who participated in the CDC Behavioral Risk Factor Surveillance System (BRFSS) survey.
*   **Time Horizon:** 2015 survey data.
*   **Features:** 21 health indicator variables, including BMI, high blood pressure, high cholesterol, physical activity, general health, age, income, and education.
*   **Target Definition:** The raw target has three classes: `0 = no diabetes`, `1 = prediabetes`, and `2 = diabetes`. In our current modeling pipeline, we convert this into a binary target: `0 = no diabetes or prediabetes`, `1 = diabetes`.

**Anti-goals:**
*   We will *not* build a clinical diagnosis tool. Clinical diagnosis requires medical testing.
*   We will *not* prescribe treatment, medication, or individual medical advice.
*   We will *not* distinguish Type I vs. Type II diabetes.
*   We will *not* separately model prediabetes in the current version; prediabetes is grouped with the non-diabetes class.

---

## 2. Data Gathering

**Source Identification:**
*   Public Kaggle dataset: "Diabetes Health Indicators Dataset", derived from the CDC BRFSS 2015 dataset.

**Acquisition Strategy:**
*   One-time download of the prepared CSV file from Kaggle. No automated scraping pipeline is used.

**Documentation of Provenance:**
*   **URL:** `https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data`
*   **Original Source:** CDC BRFSS 2015 Survey.
*   **Raw File Used:**
    *   `data/raw/diabetes_012_health_indicators_BRFSS2015.csv`
*   **Processed Files Created:**
    *   `data/processed/cleaned_diabetes_01_data.csv`
    *   `data/processed/train.csv`
    *   `data/processed/test.csv`

**Current Data Processing Pipeline:**
*   `notebooks/01_eda_and_data_cleaning.ipynb` loads the raw `012` dataset.
*   The original `Diabetes_012` target is converted to binary:
    *   original `0` and `1` become `0`
    *   original `2` becomes `1`
*   The target column is renamed from `Diabetes_012` to `Diabetes_01`.
*   The cleaned binary dataset is saved as `cleaned_diabetes_01_data.csv`.
*   A stratified 80/20 train-test split is created with `random_state=42`.
*   Model selection notebooks use `train.csv` for 5-fold cross-validation.

**Ethical and Legal Considerations:**
*   **Licensing:** The dataset is derived from a public CDC survey.
*   **Privacy:** The data is anonymized and does not include direct personally identifiable information.
*   **Ethical Handling:** The data includes sensitive social and health-related variables. Results should be interpreted carefully, especially for features related to age, income, education, and access to care.

---

## 3. Data Assessment

**Volume and Coverage:**
*   The raw dataset contains over 250,000 survey responses, which is sufficient for training and comparing several classification models.

**Granularity:**
*   Each row represents one individual respondent, matching the project unit of analysis.

**Target Construction:**
*   The original dataset supports three-class modeling, but the current project focuses on binary classification: diabetes vs. non-diabetes/prediabetes.
*   This makes the modeling task clearer for screening: identify respondents likely to belong to the diabetes class.
*   The binary target is imbalanced because diabetes cases are less common than non-diabetes/prediabetes cases.

**Bias and Representativeness:**
*   **Selection Bias:** BRFSS is a survey dataset, so it may underrepresent people who are less reachable by phone or less likely to respond.
*   **Response/Recall Bias:** BMI, physical activity, diet, and general health are self-reported or survey-derived, so they may contain reporting error.
*   **Class Imbalance:** The positive diabetes class is smaller than the negative class, so accuracy alone is not enough to evaluate model quality.

**Exploratory Data Analysis Strategy:**
*   The EDA notebook checks target distribution, feature distributions, correlations, and BMI patterns.
*   The heatmap is used to identify features most correlated with the binary diabetes target.
*   These observations are later compared with logistic regression coefficients to see whether important model features align with EDA patterns.

---

## 4. Assessing Learnability

**Signal vs. Noise:**
*   The feature set contains medically plausible signals. High blood pressure, high cholesterol, BMI, age, general health, and difficulty walking are all expected to relate to diabetes risk.

**Data Sufficiency:**
*   The full BRFSS-derived dataset provides enough examples for standard machine learning models, including linear models, tree-based models, support vector machines, gradient boosting, and neural networks.

**Feature-Target Alignment:**
*   The features are available at prediction time because they are survey-style health indicators.
*   The target is based on reported diabetes status, so the model should be understood as predicting current diabetes class from survey data, not diagnosing future disease onset.

**Models Implemented So Far:**
*   `notebooks/02_baseline.ipynb`: Dummy classifier baseline.
*   `notebooks/03_logistic_regression.ipynb`: Logistic regression, Lasso logistic regression, and Ridge logistic regression with coefficient comparison.
*   `notebooks/04_random_forest.ipynb`: Random forest with 5-fold cross-validation.
*   `notebooks/05_gradient_boost.ipynb`: XGBoost classifier with 5-fold cross-validation.
*   `notebooks/06_svm.ipynb`: Linear SVM with 5-fold cross-validation.
*   `notebooks/07_nn.ipynb`: MLP neural network using Keras/TensorFlow with 5-fold cross-validation.

---

## 5. KPI Definition (Key Performance Indicators)

**Primary KPI:**
*   **Recall (Sensitivity) for the Diabetes Class.**
    *   *Rationale:* In a screening context, false negatives are costly because a person with diabetes may not be flagged for follow-up care. Higher recall helps identify more true diabetes cases.

**Secondary KPIs:**
*   **Precision:** Measures how many predicted diabetes cases are actually diabetes cases.
*   **F1-Score:** Balances precision and recall equally.
*   **F2-Score:** Gives more weight to recall than precision, which fits the screening goal better than F1 alone.
*   **AUPRC:** Measures ranking quality under class imbalance and is useful when the positive class is relatively rare.
*   **ROC-AUC:** Measures overall discrimination between the two classes across thresholds.
*   **Predicted Positive Rate:** Helps interpret how aggressive each threshold is.

**Validation Strategy:**
*   The processed data is split into stratified train and test sets.
*   Model comparison is performed using 5-fold stratified cross-validation on the training set.
*   Metrics are computed directly from validation-fold predictions, not through `make_scorer`.

**Threshold Strategy:**
For models that output probabilities or decision scores, we compare:
1.  **Default threshold**
    *   `0.5` for probability-based models.
    *   `0.0` for the Linear SVM decision function.
2.  **Max F2 threshold**
    *   Selects the threshold that maximizes F2, prioritizing recall.
3.  **Max TPR-FPR threshold**
    *   Selects the threshold that maximizes `TPR - FPR` from the ROC curve.

**Baseline Definition:**
*   The baseline notebook uses a `DummyClassifier(strategy="most_frequent")`.
*   All stronger models are compared against this baseline using recall, F1, F2, and related validation metrics.
