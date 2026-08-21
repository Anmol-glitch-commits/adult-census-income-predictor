import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Adult Census Income Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 40px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 20px;
        border: 1px solid #ddd;
    }

    .high-income {
        background-color: #e8f5e9;
        border-left: 6px solid #2e7d32;
    }

    .low-income {
        background-color: #fff8e1;
        border-left: 6px solid #f9a825;
    }

    .probability-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #f5f7fa;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">📊 Adult Census Income Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict whether a person is likely to earn ≤50K or >50K annually.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATASET AUTOMATICALLY
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("adult_census.csv")

try:
    df = load_data().copy()

except FileNotFoundError:

    st.error(
        "❌ Dataset not found. Please make sure "
        "'adult_census.csv' is inside the 'data' folder."
    )

    st.stop()


# ============================================================
# BASIC CLEANING
# ============================================================

# Remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Strip whitespace from all object/string columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()


# ============================================================
# REPLACE '?' WITH 'Unknown'
# ============================================================

# Replace '?' with 'Unknown' in categorical columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].replace("?", "Unknown")


# ============================================================
# NORMALIZE INCOME TARGET
# ============================================================

target_column = "income"

if target_column not in df.columns:

    st.error(
        f"❌ Target column '{target_column}' was not found "
        "in the dataset."
    )

    st.stop()


# Remove possible punctuation/spacing variations
df[target_column] = (
    df[target_column]
    .astype(str)
    .str.strip()
    .replace({
        "<=50K.": "<=50K",
        ">50K.": ">50K",
        "<=50K ": "<=50K",
        ">50K ": ">50K"
    })
)


# ============================================================
# CHECK TARGET VALUES
# ============================================================

valid_targets = {"<=50K", ">50K"}

invalid_targets = (
    set(df[target_column].unique()) - valid_targets
)

if invalid_targets:

    st.error(
        f"❌ Unexpected income labels found: {invalid_targets}"
    )

    st.stop()


# ============================================================
# NOTE:
# Duplicate removal is intentionally done AFTER dropping the same
# non-model columns as the notebook (Education, Final_census,
# Unnamed: 15). This reproduces the notebook's 3,441 duplicate
# removal step more faithfully.
# ============================================================

# ============================================================
# HANDLE MISSING VALUES LIKE THE NOTEBOOK
# ============================================================

# The notebook replaces '?' with NaN and then uses an explicit
# 'Unknown' category for these categorical columns.
for col in ["Profession Class", "occupation", "country"]:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")


# ============================================================
# MAKE CATEGORY LABELS HUMAN-READABLE
# ============================================================

profession_map = {
    "Local-gov": "Local government",
    "Never-worked": "Never worked",
    "Private": "Private sector",
    "Self-emp-inc": "Self-employed (incorporated)",
    "Self-emp-not-inc": "Self-employed (not incorporated)",
    "State-gov": "State government",
    "Without-pay": "Without pay"
}

if "Profession Class" in df.columns:
    df["Profession Class"] = df["Profession Class"].replace(profession_map)


# ============================================================
# DROP UNUSED COLUMNS
# ============================================================

columns_to_drop = [
    "Final_census",
    "Education",
    "Unnamed: 15"
]

existing_columns_to_drop = [
    col
    for col in columns_to_drop
    if col in df.columns
]

if existing_columns_to_drop:
    df = df.drop(columns=existing_columns_to_drop)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

before_duplicates = len(df)

df = df.drop_duplicates()

duplicates_removed = before_duplicates - len(df)


# ============================================================
# REQUIRED FEATURES
# ============================================================

numeric_cols = [
    "Age",
    "Education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]


categorical_cols = [
    "Profession Class",
    "marital-status",
    "occupation",
    "relationship",
    "Gender",
    "country"
]


required_columns = (
    numeric_cols
    + categorical_cols
    + [target_column]
)


missing_required = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_required:

    st.error(
        "❌ The dataset is missing these required columns:"
    )

    st.write(missing_required)

    st.stop()


# ============================================================
# DATASET INFORMATION
# ============================================================

st.subheader("📋 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Rows",
        f"{df.shape[0]:,}"
    )


with col2:

    st.metric(
        "Features",
        df.shape[1] - 1
    )


with col3:

    st.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )


with col4:

    st.metric(
        "Duplicates Removed",
        duplicates_removed
    )


# ============================================================
# PREVIEW DATASET
# ============================================================

with st.expander("👀 Preview Dataset"):

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

with st.expander("📊 Income Distribution"):

    income_distribution = (
        df[target_column]
        .value_counts()
        .rename_axis("Income")
        .reset_index(name="Count")
    )

    income_distribution["Percentage"] = (
        income_distribution["Count"]
        / income_distribution["Count"].sum()
        * 100
    )

    st.dataframe(
        income_distribution,
        use_container_width=True
    )


# ============================================================
# PREPARE X AND Y
# ============================================================

X = df[
    numeric_cols + categorical_cols
].copy()

y = df[target_column].copy()


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# PREPROCESSING PIPELINES
# ============================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),

    (
        "scaler",
        StandardScaler()
    )
])


categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),

    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])


# ============================================================
# COLUMN TRANSFORMER
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[

        (
            "numeric",
            numeric_pipeline,
            numeric_cols
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_cols
        )
    ]
)


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

@st.cache_resource
def train_model(X_train, y_train):

    pipeline = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=4,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    pipeline.fit(
        X_train,
        y_train
    )

    return pipeline


# ============================================================
# TRAIN MODEL
# ============================================================

with st.spinner(
    "🤖 Training Random Forest model..."
):

    model = train_model(
        X_train,
        y_train
    )


st.success(
    "✅ Model trained successfully!"
)


# ============================================================
# MODEL EVALUATION
# ============================================================

y_test_pred = model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    y_test_pred
)


f1 = f1_score(
    y_test,
    y_test_pred,
    pos_label=">50K"
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader("📈 Model Performance")

metric1, metric2 = st.columns(2)


with metric1:

    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )


with metric2:

    st.metric(
        "F1 Score (>50K)",
        f"{f1:.4f}"
    )


# ============================================================
# MODEL PARAMETERS
# ============================================================

with st.expander("⚙️ Model Configuration"):

    st.write({
        "Algorithm": "Random Forest Classifier",
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 4,
        "max_features": "sqrt",
        "random_state": 42
    })


# ============================================================
# PREDICTION SECTION
# ============================================================

st.divider()

st.subheader("🔮 Predict Individual Income")

st.write(
    "Enter the details below and the model will "
    "estimate the income class."
)


# ============================================================
# EDUCATION MAPPING
# ============================================================

# Education-num is the numeric ordinal representation used by the model.
# The UI shows a human-readable label, then extracts the corresponding number.
education_mapping = {
    1: "Preschool",
    2: "1st-4th grade",
    3: "5th-6th grade",
    4: "7th-8th grade",
    5: "9th grade",
    6: "10th grade",
    7: "11th grade",
    8: "12th grade",
    9: "High school graduate",
    10: "Some college",
    11: "Associate degree (vocational)",
    12: "Associate degree (academic)",
    13: "Bachelor's degree",
    14: "Master's degree",
    15: "Professional school degree",
    16: "Doctorate degree"
}


education_options = [
    f"{num} - {education}"
    for num, education in education_mapping.items()
]


# ============================================================
# INPUT FORM
# ============================================================

with st.form("prediction_form"):

    left_col, right_col = st.columns(2)


    # --------------------------------------------------------
    # LEFT COLUMN
    # --------------------------------------------------------

    with left_col:

        # ----------------------------------------------------
        # AGE
        # ----------------------------------------------------

        age = st.number_input(
            "Age",
            min_value=17,
            max_value=90,
            value=30,
            step=1
        )


        # ----------------------------------------------------
        # PROFESSION CLASS
        # ----------------------------------------------------

        profession_options = sorted(
            df["Profession Class"]
            .dropna()
            .unique()
            .tolist()
        )


        profession = st.selectbox(
            "Profession Class",
            profession_options
        )


        # ----------------------------------------------------
        # EDUCATION LEVEL
        # ----------------------------------------------------

        selected_education = st.selectbox(
            "Education Level",
            education_options,
            index=8
        )


        # Extract Education-num from selection
        education_num = int(
            selected_education.split(" - ")[0]
        )


        # ----------------------------------------------------
        # MARITAL STATUS
        # ----------------------------------------------------

        marital_options = sorted(
            df["marital-status"]
            .dropna()
            .unique()
            .tolist()
        )


        marital_status = st.selectbox(
            "Marital Status",
            marital_options
        )


        # ----------------------------------------------------
        # OCCUPATION
        # ----------------------------------------------------

        occupation_options = sorted(
            df["occupation"]
            .dropna()
            .unique()
            .tolist()
        )


        occupation = st.selectbox(
            "Occupation",
            occupation_options
        )


        # ----------------------------------------------------
        # RELATIONSHIP
        # ----------------------------------------------------

        relationship_options = sorted(
            df["relationship"]
            .dropna()
            .unique()
            .tolist()
        )


        relationship = st.selectbox(
            "Relationship",
            relationship_options
        )


    # --------------------------------------------------------
    # RIGHT COLUMN
    # --------------------------------------------------------

    with right_col:

        # ----------------------------------------------------
        # GENDER
        # ----------------------------------------------------

        gender_options = sorted(
            df["Gender"]
            .dropna()
            .unique()
            .tolist()
        )


        gender = st.selectbox(
            "Gender",
            gender_options
        )


        # ----------------------------------------------------
        # CAPITAL GAIN
        # ----------------------------------------------------

        capital_gain = st.number_input(
            "Capital Gain",
            min_value=0,
            max_value=1_000_000,
            value=0,
            step=100
        )


        # ----------------------------------------------------
        # CAPITAL LOSS
        # ----------------------------------------------------

        capital_loss = st.number_input(
            "Capital Loss",
            min_value=0,
            max_value=100_000,
            value=0,
            step=100
        )


        # ----------------------------------------------------
        # HOURS PER WEEK
        # ----------------------------------------------------

        hours_per_week = st.number_input(
            "Hours per Week",
            min_value=1,
            max_value=99,
            value=40,
            step=1
        )


        # ----------------------------------------------------
        # COUNTRY
        # ----------------------------------------------------

        country_options = sorted(
            df["country"]
            .dropna()
            .unique()
            .tolist()
        )


        country = st.selectbox(
            "Country",
            country_options
        )


    st.write("")


    submitted = st.form_submit_button(
        "🚀 Predict Income",
        use_container_width=True
    )


# ============================================================
# MAKE PREDICTION
# ============================================================

if submitted:

    # --------------------------------------------------------
    # CREATE ONE-ROW INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame({

        "Age": [age],

        "Education-num": [
            education_num
        ],

        "capital-gain": [
            capital_gain
        ],

        "capital-loss": [
            capital_loss
        ],

        "hours-per-week": [
            hours_per_week
        ],

        "Profession Class": [
            profession
        ],

        "marital-status": [
            marital_status
        ],

        "occupation": [
            occupation
        ],

        "relationship": [
            relationship
        ],

        "Gender": [
            gender
        ],

        "country": [
            country
        ]
    })


    # --------------------------------------------------------
    # GET PROBABILITIES
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        input_data
    )[0]


    # --------------------------------------------------------
    # GET CLASS LABELS
    # --------------------------------------------------------

    class_names = model.classes_


    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    if len(class_names) != 2:

        st.error(
            f"❌ Expected 2 classes, but found: "
            f"{class_names}"
        )

        st.stop()


    # --------------------------------------------------------
    # CREATE PROBABILITY DICTIONARY
    # --------------------------------------------------------

    probability_dict = dict(
        zip(
            class_names,
            probabilities
        )
    )


    # --------------------------------------------------------
    # DERIVE PREDICTION FROM HIGHEST PROBABILITY
    # --------------------------------------------------------

    predicted_index = np.argmax(
        probabilities
    )


    prediction = class_names[
        predicted_index
    ]


    # --------------------------------------------------------
    # COMPARE WITH SKLEARN PREDICT()
    # --------------------------------------------------------

    sklearn_prediction = model.predict(
        input_data
    )[0]


    if prediction != sklearn_prediction:

        st.warning(
            "⚠️ predict() and predict_proba() gave "
            "different results. Please inspect "
            "the trained model."
        )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.subheader("🎯 Prediction Result")


    high_income_probability = (
        probability_dict.get(">50K", 0)
    )


    low_income_probability = (
        probability_dict.get("<=50K", 0)
    )


    # --------------------------------------------------------
    # HIGH INCOME
    # --------------------------------------------------------

    if prediction == ">50K":
        st.success(
            f"💰 Predicted Income: >50K\n\n"
            f"The model estimates a "
            f"{high_income_probability * 100:.2f}% probability "
            f"of income being above 50K."
        )


    # --------------------------------------------------------
    # LOW INCOME
    # --------------------------------------------------------

    else:
        st.info(
            f"💵 Predicted Income: ≤50K\n\n"
            f"The model estimates a "
            f"{low_income_probability * 100:.2f}% probability "
            f"of income being 50K or below."
        )


    # ========================================================
    # PROBABILITY DISPLAY
    # ========================================================

    st.subheader("📊 Prediction Probabilities")


    prob_col1, prob_col2 = st.columns(2)


    with prob_col1:

        st.metric(
            "≤50K Probability",
            f"{low_income_probability * 100:.2f}%"
        )


    with prob_col2:

        st.metric(
            ">50K Probability",
            f"{high_income_probability * 100:.2f}%"
        )


    # ========================================================
    # PROBABILITY BAR
    # ========================================================

    probability_df = pd.DataFrame({

        "Income Class": [
            "<=50K",
            ">50K"
        ],

        "Probability": [
            low_income_probability * 100,
            high_income_probability * 100
        ]
    })


    st.bar_chart(
        probability_df.set_index(
            "Income Class"
        )
    )


    # ========================================================
    # INPUT DETAILS
    # ========================================================

    with st.expander(
        "🔍 View Input Used for Prediction"
    ):

        st.dataframe(
            input_data,
            use_container_width=True
        )


    # ========================================================
    # MODEL DECISION DETAILS
    # ========================================================

    with st.expander(
        "🧠 Model Decision Details"
    ):

        st.write(
            "Classes learned by the model:"
        )

        st.write(
            list(class_names)
        )


        st.write(
            "Raw probabilities returned by Random Forest:"
        )

        st.write(
            probabilities
        )


        st.write(
            "Final prediction:"
        )

        st.write(
            prediction
        )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.divider()

st.sidebar.title(
    "📌 Project Information"
)

st.sidebar.write(
    """
    **Dataset:** Adult Census Income

    **Problem:** Binary Classification

    **Target:** Income

    **Algorithm:** Random Forest

    **Preprocessing:**
    - Unknown category for missing categorical values
    - Median Imputation
    - Standard Scaling
    - Most-Frequent Imputation
    - One-Hot Encoding
    - Exact duplicate removal after dropping unused columns

    **Hyperparameters:**
    - 500 Trees
    - min_samples_leaf = 4
    - max_features = sqrt
    - max_depth = None
    """
)


st.sidebar.divider()

st.sidebar.success(
    "✅ Adult Census dataset loaded automatically."
)


st.sidebar.caption(
    "Built with Python, Scikit-learn and Streamlit."
)
