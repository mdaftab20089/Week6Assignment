

### 📄 **README.md** – ML Model Comparison on Titanic Dataset

---

## 🧠 Project Title:

**Machine Learning Model Comparison using the Titanic Dataset**

---

## 📌 Objective:

This project aims to train, evaluate, and compare multiple machine learning models to predict passenger survival on the Titanic. Evaluation is done using performance metrics such as Accuracy, Precision, Recall, F1-score, and Confusion Matrix. Additionally, hyperparameter tuning is performed to optimize the best model.

---

## 📊 Dataset:

* **Source**: [Seaborn Titanic Dataset](https://github.com/mwaskom/seaborn-data)
* **Target**: `survived` (0 = No, 1 = Yes)
* **Features**: Age, Sex, Fare, Pclass, SibSp, Parch, Embarked, etc.

---

## 🔧 Steps Performed:

### 1. **Exploratory Data Analysis (EDA)**:

* Plotted survival counts
* Examined survival by gender
* Analyzed age and fare distributions
* Visualized feature correlation using a heatmap

### 2. **Data Preprocessing**:

* Missing value imputation for numerical and categorical features
* One-hot encoding of categorical variables
* Standard scaling of numerical features
* Train-test split (80/20)

### 3. **Models Trained**:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Support Vector Machine (SVC)
* K-Nearest Neighbors (KNN)

### 4. **Evaluation Metrics**:

Each model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix (plotted)

### 5. **Hyperparameter Tuning**:

Performed `GridSearchCV` on Random Forest with parameters:

* `n_estimators`: \[50, 100, 200]
* `max_depth`: \[4, 6, 8, None]

---

## ✅ Results:

* All models were evaluated and compared.
* The **best-performing model** was determined based on **F1-score** and validated with a **confusion matrix** and **classification report**.

---

## 📦 Dependencies:

* `pandas`, `numpy`, `seaborn`, `matplotlib`
* `scikit-learn`

Install with:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

---

## 🚀 How to Run:

1. Open the Python script or Jupyter Notebook.
2. Run all cells from top to bottom.
3. Visuals and model performance will be displayed.
4. The best model's performance is printed at the end.

---

## 🏁 Future Improvements:

* Try ensemble methods like XGBoost
* Perform cross-validation for more reliable metrics
* Deploy the model using Flask or Streamlit

---

