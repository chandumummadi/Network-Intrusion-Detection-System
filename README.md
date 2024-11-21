# **Network Intrusion Detection**

This project demonstrates the development of a **Machine Learning-based Intrusion Detection System (IDS)** using the **UNSW-NB15 dataset**. The system utilizes **Random Forest** as the primary model for detecting anomalies and potential intrusions in network traffic. This README outlines the motivation, steps, and results of the project.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Dependencies](#dependencies)
4. [Workflow](#workflow)
5. [Results](#results)
6. [Usage](#usage)
7. [Contributors](#contributors)
8. [License](#license)

---

## **Project Overview**
The goal of this project is to develop a robust Intrusion Detection System to classify network traffic as either **normal** or **malicious**. It focuses on:
- **Feature Engineering** to extract meaningful insights from raw network data.
- **Modeling and Evaluation** of machine learning models, with emphasis on Random Forest.
- **Optimization** of models using hyperparameter tuning.
- Comparative analysis of **Random Forest**, **XGBoost**, and **LightGBM** to choose the best-performing model.

---

## **Dataset**
The **UNSW-NB15 dataset** is used for training and testing. This dataset contains both normal and attack traffic, making it ideal for supervised learning tasks.

- **Link to Dataset**: [UNSW-NB15 Dataset on Kaggle](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)
- **Training Data**: Contains labeled examples for training.
- **Test Data**: Separate labeled examples for evaluation.

---

## **Dependencies**
Ensure you have the following libraries installed:

```bash
pandas==1.x.x
numpy==1.x.x
scikit-learn==1.x.x
xgboost==1.x.x
lightgbm==3.x.x
matplotlib==3.x.x
seaborn==0.x.x
```

Install them using:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn
```

---

## **Workflow**
### 1. **Exploratory Data Analysis (EDA):**
   - Data preprocessing (handling missing values, encoding categorical variables).
   - Visualizations to understand feature distributions and correlations.
   - Feature selection using methods like:
     - Feature Importance from Random Forest
     - Chi-Square Test for categorical features.

### 2. **Model Training and Evaluation:**
   - Tried **Random Forest**, **XGBoost**, and **LightGBM** models.
   - Hyperparameter tuning using `GridSearchCV`.
   - Compared models based on accuracy, F1-score, and confusion matrices.

### 3. **Results and Selection:**
   - Selected **Random Forest** for its balanced performance across metrics.
   - Results before and after tuning were analyzed to finalize the model.

---

## **Results**
- **Selected Model**: Random Forest
- **Performance Metrics**:
  - **Accuracy**: 91%
  - **F1-Score**: 0.90 (macro average)
  - **Precision (Class 1)**: 98%
  - **Recall (Class 1)**: 88%
- **Feature Importance**:
  - Key features identified include:
    - `sload`
    - `rate`
    - `ct_dst_sport_ltm`
    - `sttl`
    - `attack_cat_3`

---

## **Usage**
### **Steps to Run the Project:**

1. Clone the repository:
   ```bash
   git clone https://github.com/chandumummadi/Network-Intrusion-Detection-System.git
   ```
2. Navigate to the project folder:
   ```bash
   cd Network-Intrusion-Detection
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
5. Open the main project file and execute the cells to:
   - Perform EDA
   - Train models
   - Evaluate the results

---

## **Contributors**
- **Sharath Chandra Mummadi**
- Special thanks to the creators of the **UNSW-NB15 dataset**.

---

## **License**
This project is free to use, modify, and distribute for any purpose. No attribution is required, but it's appreciated.
