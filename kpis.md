# KPI Definition and Modeling Strategy

This file documents the current evaluation strategy for the diabetes prediction project.

## Target Definition

The raw dataset uses a three-class target:

* `0`: no diabetes
* `1`: prediabetes
* `2`: diabetes

In the current pipeline, `notebooks/01_eda_and_data_cleaning.ipynb` converts this into a binary classification target:

* `0`: no diabetes or prediabetes
* `1`: diabetes

The target column is renamed from `Diabetes_012` to `Diabetes_01`.

## Primary KPI

* **Recall / Sensitivity for the diabetes class (`1`)**

Rationale: this project is framed as a screening problem. A false negative means the model fails to flag a person with diabetes for follow-up testing or care. In this context, missing a true diabetes case is more costly than incorrectly flagging someone who may later receive a confirmatory clinical test.

## Secondary KPIs

* **Precision:** Measures how many predicted diabetes cases are actually diabetes cases. This helps monitor the cost of false positives.
* **F1-score:** Balances precision and recall equally.
* **F2-score:** Gives more weight to recall than precision, which fits the screening goal better than F1 alone.
* **AUPRC:** Measures ranking quality for the positive class and is especially useful under class imbalance.
* **AUROC:** Measures the model's ability to separate diabetes and non-diabetes/prediabetes cases across thresholds.
* **Predicted positive rate:** Shows how many people the model would flag under a chosen threshold.
* **Accuracy:** Useful for baseline context, but not sufficient for model selection because the target is imbalanced.

## Validation Strategy

The cleaned dataset is split into:

* `data/processed/train.csv`
* `data/processed/test.csv`

The train/test split is stratified with `test_size=0.20` and `random_state=42`.

Current model comparison is performed using **5-fold stratified cross-validation on the training set**. Validation metrics are computed directly from each fold's validation predictions, not with `make_scorer`.

The test set should be reserved for final evaluation after model selection is complete.

## Baseline Definition

The baseline model is:

* **Dummy classifier:** `DummyClassifier(strategy="most_frequent")`

The dummy model mostly predicts the majority class, so its accuracy can look high even though it does not identify diabetes cases well. This baseline is mainly used to show why recall, F2, AUPRC, and AUROC are more informative than accuracy alone.

## Models Evaluated So Far

The current model-selection notebooks include:

1. **Baseline:** Dummy classifier
2. **Logistic regression:** regular logistic regression, Lasso logistic regression, and Ridge logistic regression
3. **Random forest**
4. **XGBoost**
5. **Linear SVM**
6. **MLP neural network:** Keras/TensorFlow feed-forward neural network

## Threshold Strategy

For models that output probabilities or decision scores, model behavior is summarized under three threshold choices:

1. **Default threshold**
   * `0.5` for probability-based models
   * `0.0` for the Linear SVM decision function
2. **Max F2 threshold**
   * Selects the threshold that maximizes F2
   * Prioritizes recall for screening
3. **Max TPR-FPR threshold**
   * Selects the ROC threshold that maximizes `TPR - FPR`
   * Provides a middle-ground operating point between sensitivity and false positive rate

Threshold selection should be treated as a business, public health, or medical operations decision. A lower threshold can improve recall, but it will also increase the number of people flagged for follow-up.

## Current Model Selection Direction

Based on cross-validation results so far:

* **XGBoost** has the strongest overall ranking metrics among the current models.
* **MLP neural network** is competitive with XGBoost, especially on recall and F2.
* **Logistic regression** performs close to more complex methods and is the most interpretable model.
* **Linear SVM** can achieve very high recall under the max-F2 threshold, but precision drops.
* **Random forest** is more conservative at the default threshold, giving higher precision but much lower recall.

For the final report, we should compare performance-driven selection against interpretability-driven selection. XGBoost is a strong candidate for predictive performance, while logistic regression is useful for explaining feature effects.
