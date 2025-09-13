# Customer Churn Prediction (SVM vs Decision Tree)

## 📌 Overview

This project predicts whether a customer will **churn (leave a service)** based on billing and usage data.
It compares two models: **Support Vector Machine (SVM)** and **Decision Tree Classifier**.

---

## 📊 Dataset

* **File:** `customer_churn_tree_svm (1).csv`
* **Shape:** 150 rows × 5 columns
* **Features:**

  * `monthly_charges` – Monthly billing amount
  * `tenure_months` – Number of months subscribed
  * `contract_type` – Type of contract (0,1,2 encoding)
  * `support_calls` – Number of support calls made
* **Target:** `churn` (0 = no churn, 1 = churn)

---

## 🔎 Exploratory Data Analysis

* No missing values or duplicates.
* Skewness checked → churn column is highly skewed.
* Histograms & boxplots show feature distributions.

---

## ⚙️ Modeling

* **Train-Test Split:** 75% train, 25% test (stratified).
* **Scaling:** StandardScaler applied to features for SVM.
* **Models Used:**

  * SVM (RBF kernel, probability=True)
  * Decision Tree (default params)

---

## 📈 Results

Both models achieved **perfect performance (100%)** on the dataset:

| Model            | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ---------------- | -------- | --------- | ------ | -------- | ------- |
| **SVM**          | 1.0      | 1.0       | 1.0    | 1.0      | 1.0     |
| **DecisionTree** | 1.0      | 1.0       | 1.0    | 1.0      | 1.0     |

* **Confusion Matrices** plotted for both models.
* **ROC Curves** show perfect separation.

---

## 🔮 Prediction Example

For a new customer:

```python
monthly_charges = 55
tenure_months = 24
contract_type = 2
support_calls = 1
```

✅ Both SVM & Decision Tree predict: **0 → No Churn**

---

## 🚀 How to Run

1. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

2. Place dataset in working directory.
3. Run the Jupyter Notebook / Python script.

---

⚠️ **Note:** The dataset is small (150 rows).
That’s why both models show **overfitting/perfect accuracy**.
For real-world usage, a larger dataset is needed.

---
