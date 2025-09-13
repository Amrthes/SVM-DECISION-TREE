# Customer Churn Prediction 📊

This project predicts customer churn using **Support Vector Machine (SVM)** and **Decision Tree** models.
It is built with **Python, Scikit-learn, and Streamlit (for app deployment)**.

---

## 🚀 Features

* Preprocessed telecom dataset for churn prediction
* Trained models:

  * Decision Tree (`dt_model.pkl`)
  * Support Vector Machine (`svm_model.pkl`)
* Scaler (`scaler.pkl`) for consistent input transformation
* Streamlit app (`app.py`) for interactive prediction
* Training script (`train_and_save_models.py`)

---

## 📂 Project Structure

```
customer-churn/
│── app.py                  # Streamlit app for predictions
│── train_and_save_models.py # Script to train and save models
│── dt_model.pkl            # Saved Decision Tree model
│── svm_model.pkl           # Saved SVM model
│── scaler.pkl              # Saved Scaler
│── model.sav               # Backup model file
│── requirements.txt        # Project dependencies
│── main.ipynb              # Jupyter notebook (EDA & training)
│── README.md               # Project documentation
```

---

## ⚙️ Installation

1. Clone this repository:

```bash
git clone https://github.com/Amrthes/SVM-DECISION-TREE.git
cd SVM-DECISION-TREE
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate   # For Windows
source venv/bin/activate  # For Linux/Mac
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train models (if you want to retrain):

```bash
python train_and_save_models.py
```

### Run the Streamlit app:

```bash
streamlit run app.py
```

Then open the local URL (default: [http://localhost:8501](http://localhost:8501)) in your browser.

---

## 📊 Dataset

The dataset used includes customer information such as:

* **Monthly Charges**
* **Tenure (Months)**
* **Contract Type**
* **Support Calls**
* **Churn (0 = No, 1 = Yes)**

---

## 📌 Future Improvements

* Add more ML models (Random Forest, XGBoost)
* Improve feature engineering
* Deploy on **Heroku/Render/Streamlit Cloud**

---

## 🧑‍💻 Author

👤 **Amr**
[GitHub](https://github.com/Amrthes)

---
