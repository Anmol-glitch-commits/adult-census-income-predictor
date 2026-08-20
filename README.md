# 💵 Adult Census Income Predictor

A machine learning web application that predicts whether an individual's annual income is **≤50K or >50K** based on demographic, educational, occupational, and financial attributes from the Adult Census dataset.

The project covers the complete machine learning workflow — from **data preprocessing and feature engineering to model training, evaluation, and deployment using Streamlit**.

🚀 **Live Demo:** `YOUR_STREAMLIT_APP_URL`

💻 **GitHub:** `YOUR_GITHUB_REPOSITORY_URL`

---

## 📌 Project Overview

Income prediction is a binary classification problem where the objective is to determine whether a person's annual income falls into one of two categories:

* `<=50K`
* `>50K`

The model learns patterns from demographic, education, employment, working-hour, and financial information to make predictions for new individuals.

The trained model is integrated into a **Streamlit web application**, allowing users to enter individual details through an interactive interface and receive an income prediction along with the model's confidence/probability.

---

## 🎯 Problem Statement

The objective of this project is to build a machine learning classification system that can predict an individual's income category using information such as:

* Age
* Education
* Occupation
* Working hours per week
* Capital gain
* Capital loss
* Other relevant census attributes

The project focuses on building a reliable preprocessing and machine learning pipeline that can handle both **numerical and categorical features**.

---

## 📊 Dataset

The project uses the **Adult Census Income Dataset**, which contains demographic and employment-related information about individuals.

### Target Variable

The target variable represents the individual's income category:

| Target  | Meaning                       |
| ------- | ----------------------------- |
| `<=50K` | Annual income is 50K or below |
| `>50K`  | Annual income is above 50K    |

---

## 🧾 Features

The dataset contains a mixture of numerical and categorical features.

### Numerical Features

Examples include:

* Age
* Education Number
* Capital Gain
* Capital Loss
* Hours per Week

### Categorical Features

Examples include:

* Workclass
* Education
* Marital Status
* Occupation
* Relationship
* Race
* Sex
* Native Country

The final feature set used by the application is defined in `app.py`.

---

## 🔄 Machine Learning Workflow

The project follows the following machine learning pipeline:

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
     │
     ├───────────────┐
     │               │
     ▼               ▼
Numerical        Categorical
Features         Features
     │               │
     ▼               ▼
Imputation       Imputation
     │               │
     ▼               ▼
StandardScaler   OneHotEncoder
     │               │
     └───────┬───────┘
             ▼
     ColumnTransformer
             │
             ▼
     Random Forest
       Classifier
             │
             ▼
        Prediction
             │
             ▼
      Streamlit App
             │
             ▼
     Income Prediction
```

---

## 🧹 Data Preprocessing

A major focus of the project is creating a consistent preprocessing pipeline for numerical and categorical data.

### Numerical Features

Numerical features are processed using:

**1. Missing Value Imputation**

`SimpleImputer` is used to handle missing values.

**2. Feature Scaling**

`StandardScaler` is used to standardize numerical features.

---

### Categorical Features

Categorical features are processed using:

**1. Missing Value Imputation**

Missing categorical values are handled using `SimpleImputer`.

**2. One-Hot Encoding**

`OneHotEncoder` converts categorical variables into numerical representations that can be used by the machine learning model.

---

## 🏗️ Preprocessing Pipeline

The preprocessing operations are implemented using Scikit-learn's:

* `Pipeline`
* `ColumnTransformer`
* `SimpleImputer`
* `StandardScaler`
* `OneHotEncoder`

This keeps preprocessing and model prediction consistent between training and inference.

It also helps prevent accidental inconsistencies between the transformations applied during model training and those applied to new user input.

---

## 🤖 Machine Learning Model

### Random Forest Classifier

The primary classification algorithm used in this project is **Random Forest**.

Random Forest is an ensemble learning algorithm that combines multiple decision trees and aggregates their predictions.

### Why Random Forest?

Random Forest was selected because it:

* Handles nonlinear relationships well
* Works effectively with mixed feature types after preprocessing
* Is relatively robust to noise
* Can capture feature interactions
* Generally performs well on tabular classification problems
* Provides probability estimates for predictions

---

## 📈 Model Evaluation

The model is evaluated using classification metrics including:

* Accuracy
* F1 Score

### Results

> **Add the final metrics obtained from your notebook here.**

| Metric   |    Score |
| -------- | -------: |
| Accuracy | `XX.XX%` |
| F1 Score |  `XX.XX` |

The F1 score is particularly useful when evaluating classification performance because it balances **precision and recall**.

---

## 🌐 Streamlit Application

The trained machine learning workflow is integrated into a Streamlit application.

Users can enter the required attributes through the web interface.

The application then:

```text
User Input
    ↓
Input Validation
    ↓
DataFrame Creation
    ↓
Preprocessing Pipeline
    ↓
Random Forest Model
    ↓
Prediction
    ↓
Probability
    ↓
Result Display
```

The application displays:

### 💵 Predicted Income

Either:

```text
<=50K
```

or:

```text
>50K
```

along with the model's estimated probability.

---

## 🖥️ Application Features

* Interactive Streamlit interface
* User-friendly input form
* Numerical and categorical input handling
* Automated preprocessing
* Random Forest based prediction
* Income classification
* Prediction probability
* Clean result presentation

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Random Forest
* Pipeline
* ColumnTransformer

### Data Processing

* Pandas
* NumPy

### Web Application

* Streamlit

### Development

* Jupyter Notebook
* VS Code
* Git
* GitHub

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
├── .gitignore
│
└── screenshots/
    ├── home.png
    ├── input.png
    └── prediction.png
```

### File Description

| File                        | Description                                          |
| --------------------------- | ---------------------------------------------------- |
| `app.py`                    | Streamlit application and prediction pipeline        |
| `adult_census.csv`          | Dataset used by the application                      |
| `AdultCensus_Project.ipynb` | EDA, preprocessing, model development and evaluation |
| `requirements.txt`          | Python dependencies                                  |
| `README.md`                 | Project documentation                                |
| `.gitignore`                | Files excluded from Git                              |

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
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

Run:

```bash
streamlit run app.py
```

The application will open in your browser at:

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
GitHub Repository
       ↓
Streamlit Community Cloud
       ↓
Public Web Application
```

### Live Application

🚀 **[Open Live Demo](YOUR_STREAMLIT_APP_URL)**

---

## 📸 Application Preview

### Home Page

*Add screenshot here.*

```text
screenshots/home.png
```

### Input Form

*Add screenshot here.*

```text
screenshots/input.png
```

### Prediction Result

*Add screenshot here.*

```text
screenshots/prediction.png
```

---

## 🔍 Key Machine Learning Concepts Demonstrated

This project demonstrates practical understanding of:

* Supervised Learning
* Binary Classification
* Train-Test Split
* Missing Value Imputation
* Numerical Feature Scaling
* Categorical Feature Encoding
* One-Hot Encoding
* ColumnTransformer
* Scikit-learn Pipelines
* Random Forest Classification
* Model Evaluation
* Accuracy
* F1 Score
* Prediction Probability
* Model Inference
* Streamlit Application Development
* Machine Learning Deployment

---

## 💡 Why Use a Pipeline?

The preprocessing and model are combined into a single Scikit-learn pipeline.

This provides a consistent workflow:

```text
Raw Input
   ↓
Preprocessing
   ↓
Feature Transformation
   ↓
Model
   ↓
Prediction
```

Instead of manually preprocessing training data and user input separately, the same transformation logic can be applied consistently during prediction.

---

## 🔮 Future Improvements

Potential improvements include:

* Hyperparameter tuning
* Model comparison with Logistic Regression, XGBoost and other classifiers
* Cross-validation
* Feature importance visualization
* Improved handling of class imbalance
* Model explainability using SHAP
* Better input validation
* Automated model retraining
* CI/CD integration
* Containerization using Docker

---

## ⚠️ Disclaimer

This project is intended for **educational and demonstration purposes**.

Income prediction is influenced by many factors, and a machine learning model trained on historical census data may contain biases or limitations. Predictions should therefore not be treated as definitive assessments of an individual's actual income.

---

## 👨‍💻 Author

**Anmol**

AI / Machine Learning Enthusiast

### Connect

* 💻 GitHub: `YOUR_GITHUB_PROFILE_URL`
* 🔗 LinkedIn: `YOUR_LINKEDIN_PROFILE_URL`

---

## ⭐ If You Found This Project Useful

If you found this project interesting, consider giving the repository a ⭐ on GitHub.
