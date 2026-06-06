## Primary KPI:
- **Recall (Sensitivity) for the Positive Class (Diabetes/Prediabetes).** Rationale: In a public health screening context, a false negative (failing to identify someone at risk of diabetes) is highly detrimental, as the disease will progress unmanaged. A false positive (flagging a healthy person as high risk) merely results in a recommendation to consult a doctor or get a routine blood test, which carries a much lower cost. Therefore, capturing as many true positives as possible is our primary metric for success.
## Secondary KPIs:
- **F1-Score:** To ensure precision doesn't drop so low that the model becomes useless (i.e., we don't want to just predict "Diabetes" for everyone). F1 provides a balance between Recall and Precision.
- **ROC-AUC (Area Under the Receiver Operating Characteristic Curve):** To measure the model's overall ability to discriminate between the healthy and diabetic/prediabetic classes across different thresholds.
- **Accuracy:** Useful as a baseline metric, particularly when using the 50/50 balanced dataset.
## Baseline Definition
### Baseline Definition & Modeling Strategy:
As part of our initial modeling, we will establish baseline performance using:
1. **Trivial Baseline:** `DummyClassifier` (predicting the most frequent class or stratified random).
2. **Linear Model:** `LogisticRegression` (good for interpretability of health risk factors).
3. **Tree-Based Model:** `RandomForestClassifier`.
### Advanced Classification Methods:
Beyond the simple baselines, our strategy involves experimenting with a diverse set of classification algorithms to find the optimal predictive model. We plan to train, tune, and systematically compare multiple methods, including:
1. **Gradient Boosting Machines** (e.g., XGBoost, LightGBM)
2. **Support Vector Machines (SVM)**
3. **K-Nearest Neighbors (KNN)**
4. **Neural Networks / Multilayer Perceptrons (MLP)** (if deemed necessary)
All models will be evaluated using K-Fold cross-validation, and we will benchmark their performance against our Primary (Recall) and Secondary (F1, ROC-AUC) KPIs.