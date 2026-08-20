# 💵 Adult Census Income Predictor

A machine learning web application that predicts whether an individual's annual income is **≤50K or >50K** using demographic, educational, occupational, and financial attributes from the Adult Census dataset.

The project demonstrates an end-to-end machine learning workflow including **data cleaning, feature selection, preprocessing, model comparison, hyperparameter tuning, evaluation, and deployment using Streamlit**.

🚀 **Live Demo:**
https://adult-census-income-predictor-n2jmng9kmqmtvrsh3zmxjy.streamlit.app/

💻 **GitHub Repository:**
https://github.com/Anmol-glitch-commits/adult-census-income-predictor

---

## 📌 Project Overview

Income prediction is a **binary classification problem** where the goal is to predict whether an individual's annual income belongs to one of two categories:

* `<=50K`
* `>50K`

The model learns patterns from demographic, employment, education-related, working-hour, and financial attributes.

The trained machine learning pipeline is integrated into a **Streamlit web application**, allowing users to enter individual information and receive:

* Predicted income category
* Prediction probability
* Model performance information
* Dataset overview

---

## 🎯 Problem Statement

The objective of this project is to build a machine learning classification system capable of predicting an individual's income category based on available census information.

The project focuses on handling:

* Numerical features
* Categorical features
* Missing values
* High-cardinality categorical variables
* Feature preprocessing
* Model selection
* Hyperparameter tuning
* Model evaluation
* Real-time inference

---

## 📊 Dataset

The project uses the **Adult Census Income Dataset**, containing demographic and employment-related information about individuals.

The original dataset contains:

* **32,561 rows**
* **16 columns**

The target variable is `income`.

### Target Variable

| Target  | Meaning                       |
| ------- | ----------------------------- |
| `<=50K` | Annual income is 50K or below |
| `>50K`  | Annual income is above 50K    |

---

## 🧾 Features Used

After data cleaning and feature selection, the model uses **11 predictor features**.

### Numerical Features

* `Age`
* `Education-num`
* `capital-gain`
* `capital-loss`
* `hours-per-week`

### Categorical Features

* `Profession Class`
* `marital-status`
* `occupation`
* `relationship`
* `Gender`
* `country`

### Removed Features

The following columns were excluded from the final model:

* `Final_census`
* `Education`
* `Unnamed: 15`

`Unnamed: 15` was an unnecessary column, while `Education` was excluded because `Education-num` provides the corresponding ordinal educational information.

---

## 🔄 Machine Learning Workflow

```text
                Raw Dataset
                     │
                     ▼
              Data Cleaning
                     │
                     ▼
             Feature Selection
                     │
                     ▼
              Train-Test Split
                     │
                     ▼
             Preprocessing
              ┌──────┴──────┐
              ▼             ▼
         Numerical      Categorical
          Features       Features
              │             │
              ▼             ▼
         Median           Most-Frequent
        Imputation        Imputation
              │             │
              ▼             ▼
       StandardScaler   OneHotEncoder
              │             │
              └──────┬──────┘
                     ▼
             ColumnTransformer
                     │
                     ▼
              Random Forest
                     │
                     ▼
          Hyperparameter Tuning
                     │
                     ▼
               Evaluation
                     │
                     ▼
             Streamlit App
                     │
                     ▼
              Prediction
```

---

## 🧹 Data Cleaning

Several data-cleaning steps are performed before model training.

### 1. Column Name Cleaning

Leading and trailing spaces are removed from column names.

### 2. Missing Value Detection

The dataset uses `?` to represent missing values. These values are converted into `NaN`.

### 3. String Cleaning

Leading and trailing whitespace is removed from categorical values.

### 4. Target Normalization

Income labels are normalized to:

```text
<=50K
>50K
```

### 5. Duplicate Removal

Duplicate rows are removed before model training.

### 6. Irrelevant Feature Removal

Unused columns are removed from the final feature set.

---

## ⚙️ Data Preprocessing

A separate preprocessing pipeline is created for numerical and categorical features.

### Numerical Pipeline

Numerical features are processed using:

1. **Median Imputation**
2. **StandardScaler**

```text
Numerical Features
       ↓
Median Imputation
       ↓
StandardScaler
```

Median imputation is used because it is less sensitive to extreme values than mean imputation.

### Categorical Pipeline

Categorical features are processed using:

1. **Most-Frequent Imputation**
2. **One-Hot Encoding**

```text
Categorical Features
       ↓
Most-Frequent Imputation
       ↓
One-Hot Encoding
```

`handle_unknown="ignore"` is used so that unseen categories during inference do not cause the application to fail.

---

## 🏗️ ColumnTransformer & Pipeline

Scikit-learn's `Pipeline` and `ColumnTransformer` are used to combine preprocessing and model training into a single workflow.

This ensures that the same preprocessing logic is applied during:

* Training
* Testing
* Streamlit inference

This reduces the risk of training-serving inconsistencies and preprocessing errors.

---

## 🤖 Model Selection

Multiple classification approaches were explored during development, including Logistic Regression and Random Forest.

Random Forest performed better because it can capture **non-linear relationships and feature interactions** that a linear decision boundary may not capture effectively.

Therefore, Random Forest was selected as the final model.

---

## 🌲 Random Forest Classifier

The final model is a **Random Forest Classifier**.

Random Forest is an ensemble learning algorithm that combines multiple decision trees and aggregates their predictions.

### Why Random Forest?

Random Forest was selected because it:

* Handles non-linear relationships
* Captures feature interactions
* Performs well on tabular data
* Is relatively robust to noise
* Works well with high-dimensional one-hot encoded features
* Provides class probability estimates
* Generally requires less feature engineering than many linear models

---

## 🔧 Hyperparameter Tuning

Hyperparameter tuning was performed using **RandomizedSearchCV** with:

* **3-fold cross-validation**
* **10 parameter combinations**
* **F1 score** as the optimization metric
* Parallel processing using `n_jobs=-1`

The parameters explored were:

```text
n_estimators:
[200, 300, 500]

max_depth:
[None, 10, 20]

min_samples_leaf:
[1, 2, 4]
```

### Final Parameters

| Parameter          | Value |
| ------------------ | ----: |
| `n_estimators`     |   500 |
| `max_depth`        |  None |
| `min_samples_leaf` |     4 |
| `max_features`     |  sqrt |
| `random_state`     |    42 |

The final configuration uses **500 trees** with a minimum of **4 samples per leaf**.

---

## 📈 Model Evaluation

The final Random Forest model was evaluated on the held-out test set.

### Results

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **87.15%** |
| Precision | **79.71%** |
| Recall    | **62.63%** |
| F1 Score  | **70.14%** |
| ROC-AUC   | **92.31%** |

### Metric Interpretation

**Accuracy — 87.15%**

The model correctly classifies approximately 87% of the test samples.

**Precision — 79.71%**

Among the individuals predicted as `>50K`, approximately 79.71% actually belong to the `>50K` class.

**Recall — 62.63%**

The model identifies approximately 62.63% of the actual `>50K` individuals.

**F1 Score — 70.14%**

F1 score provides a balance between precision and recall and was also used as the optimization metric during hyperparameter tuning.

**ROC-AUC — 92.31%**

The high ROC-AUC indicates strong overall class-separation ability across different classification thresholds.

---

## 🌐 Streamlit Application

The trained preprocessing and machine learning pipeline is integrated into a Streamlit application.

The application follows this inference workflow:

```text
User Input
    ↓
Input Validation
    ↓
DataFrame Creation
    ↓
Preprocessing Pipeline
    ↓
Random Forest
    ↓
Prediction
    ↓
Prediction Probability
    ↓
Result Display
```

The application predicts either:

```text
<=50K
```

or:

```text
>50K
```

and also displays the model's estimated probability.

---

## 🖥️ Application Features

* Interactive Streamlit interface
* User-friendly input form
* Numerical and categorical input handling
* Automatic data preprocessing
* Missing-value handling
* One-hot encoding
* Random Forest classification
* Prediction probability
* Model performance display
* Dataset overview
* Income distribution information
* Responsive web interface

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Random Forest
* Logistic Regression
* Pipeline
* ColumnTransformer
* RandomizedSearchCV

### Web Application

* Streamlit

### Development Tools

* Jupyter Notebook
* VS Code
* Git
* GitHub

### Deployment

* Streamlit Community Cloud

---

## 📁 Project Structure

```text
adult-census-income-predictor/
│
├── app.py
├── adult_census.csv
├── AdultCensus_Project.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

### File Description

| File                        | Description                                                                  |
| --------------------------- | ---------------------------------------------------------------------------- |
| `app.py`                    | Streamlit application, preprocessing pipeline, model training and prediction |
| `adult_census.csv`          | Adult Census dataset used by the application                                 |
| `AdultCensus_Project.ipynb` | Data analysis, EDA, preprocessing, model comparison, tuning and evaluation   |
| `requirements.txt`          | Python dependencies required to run the application                          |
| `README.md`                 | Project documentation                                                        |
| `.gitignore`                | Files excluded from version control                                          |

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Anmol-glitch-commits/adult-census-income-predictor.git
```

### 2. Navigate to the Project

```bash
cd adult-census-income-predictor
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### macOS / Linux

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application Locally

Run the following command:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 🚀 Deployment

The application is deployed using **Streamlit Community Cloud**.

Deployment workflow:

```text
Local Development
       ↓
Git
       ↓
GitHub
       ↓
Streamlit Community Cloud
       ↓
Public Web Application
```

### 🌐 Live Application

**https://adult-census-income-predictor-n2jmng9kmqmtvrsh3zmxjy.streamlit.app/**

---

## 💡 Key Machine Learning Concepts Demonstrated

This project demonstrates practical understanding of:

* Supervised Learning
* Binary Classification
* Data Cleaning
* Feature Selection
* Train-Test Split
* Stratified Sampling
* Missing Value Imputation
* Feature Scaling
* One-Hot Encoding
* ColumnTransformer
* Scikit-learn Pipelines
* Logistic Regression
* Random Forest
* Hyperparameter Tuning
* RandomizedSearchCV
* Cross-Validation
* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Prediction Probability
* Model Inference
* Streamlit Application Development
* Machine Learning Deployment

---

## 💡 Why Use a Pipeline?

The preprocessing steps and machine learning model are combined into a single pipeline:

```text
Raw Input
    ↓
Imputation
    ↓
Scaling / Encoding
    ↓
Feature Transformation
    ↓
Random Forest
    ↓
Prediction
```

This provides a consistent preprocessing workflow between model training and real-world inference.

It also reduces the possibility of accidentally applying different transformations to training data and user input.

---

## 🔮 Future Improvements

Possible future improvements include:

* Model explainability using SHAP
* Feature importance visualization in the Streamlit application
* Probability-threshold optimization
* Improved class-imbalance handling
* More extensive model comparison
* Automated model retraining
* Docker containerization
* CI/CD integration
* Model monitoring after deployment

---

## ⚠️ Disclaimer

This project is intended for **educational and demonstration purposes**.

Income prediction is influenced by many factors, and models trained on historical census data may contain biases and limitations. Predictions should therefore not be treated as definitive assessments of an individual's actual income.

---

## 👨‍💻 Author

**Anmol**

AI / Machine Learning Enthusiast

### Connect

* 💻 GitHub: https://github.com/Anmol-glitch-commits
* 🔗 LinkedIn: https://www.linkedin.com/in/anmol-chhabra-36670433a/

---

## ⭐ Support

If you found this project useful or interesting, consider giving the repository a ⭐ on GitHub.

