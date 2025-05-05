# 🔐 **Network Intrusion Detection System (IDS) – ML Model**

This project demonstrates the development of a **Machine Learning-based Intrusion Detection System (IDS)** using the **UNSW-NB15 dataset**. The system leverages classification algorithms—particularly **Random Forest**, **XGBoost**, and **LightGBM**—to identify malicious network traffic. This README outlines the motivation, steps, and results of the project.

---

## 📑 **Table of Contents**

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Dependencies](#dependencies)
4. [Workflow](#workflow)
5. [Results](#results)
6. [Usage](#usage)
7. [Contributors](#contributors)
8. [License](#license)

---

## 🚀 **Project Overview**

The goal of this project is to build a robust Intrusion Detection System that can classify network traffic as either **normal** or **malicious** in near-real time. Key focuses include:

* **Feature Engineering** from raw network traffic features.
* **Model Training & Hyperparameter Tuning** for high-accuracy detection.
* **Comparative Analysis** across multiple ML models to determine the best performer.
* **Scalability** for future real-time or production-grade deployment.

---

## 📂 **Dataset**

The project uses the **UNSW-NB15** dataset, a modern and well-balanced dataset for network security research.

* 📌 **Link**: [UNSW-NB15 Dataset on Kaggle](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)
* 📊 Includes labeled training and test sets with normal and various attack types.
* 📋 Features include protocol types, connection flags, port numbers, and flow statistics.

---

## 🧩 **Dependencies**

Install the required packages using:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn joblib
```

---

## 🔁 **Workflow**

### 1. 📊 **Exploratory Data Analysis (EDA)**

* Handled missing values and irrelevant columns.
* Explored numerical/categorical features with visualizations.
* Analyzed class imbalance and correlation with the target variable.

### 2. 🧠 **Feature Engineering**

* Used Label Encoding and One-Hot Encoding on categorical columns.
* Selected features using:

  * Random Forest importance
  * Chi-Squared test
  * Correlation with label

### 3. 🤖 **Model Training and Tuning**

* Trained:

  * Random Forest
  * XGBoost
  * LightGBM
* Applied **GridSearchCV** to tune:

  * `max_depth`, `n_estimators`, `min_samples_split`, etc.
* Evaluated using:

  * Accuracy
  * F1-score
  * Confusion matrix
  * Precision-Recall AUC

---

## 🏆 **Results**

### ✅ **Best Model: Tuned Random Forest**

| Metric                  | Value |
| ----------------------- | ----- |
| **Accuracy**            | 91%   |
| **F1-Score**            | 0.91  |
| **Precision (Class 1)** | 98%   |
| **Recall (Class 1)**    | 88%   |

> Random Forest provided the best tradeoff between precision and recall, with the lowest false positive rate and high generalization.

### 🔍 **Top Features**

* `sload`, `rate`, `ct_dst_sport_ltm`, `sttl`, `attack_cat_3`

---

## 🛠️ **Usage**

### 🔹 **Running the Project**

1. Clone the repository:

   ```bash
   git clone https://github.com/chandumummadi/Network-Intrusion-Detection-System.git
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Open and run the notebook in Jupyter:

   ```bash
   jupyter notebook
   ```

### 🔹 **Using the Saved Model**

After running the notebook, you’ll get two output files:

* `rf_ids_model.joblib` — the trained Random Forest model
* `final_features.pkl` — list of selected input features

Here’s how to reuse it:

```python
import joblib
import pandas as pd

# Load model and features
model = joblib.load('rf_ids_model.joblib')
features = joblib.load('final_features.pkl')

# Prepare a sample packet (replace with real values)
sample = pd.DataFrame([{
    'rate': 0.0, 'sttl': 30, 'sload': 512.0, 'ct_state_ttl': 3, 'ct_dst_sport_ltm': 1,
    'attack_cat_1': 0, 'attack_cat_2': 0, 'attack_cat_3': 1,
    'state_1': 0, 'state_2': 1, 'proto_1': 0, 'proto_2': 0, 'proto_3': 1
}])[features]

# Predict
print("Prediction:", model.predict(sample))  # 0 = Normal, 1 = Intrusion
```

---

## 👥 **Contributors**

* **Sharath Chandra Mummadi**

Special thanks to the UNSW Canberra Cyber team for making this dataset publicly available.

---

## 📜 **License**

This project is open-source and free to use. Feel free to modify or integrate into larger systems. Attribution is appreciated but not required.
