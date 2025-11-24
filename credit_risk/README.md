# Credit Risk Prediction App

A machine learning application that predicts credit risk (Good or Bad) for loan applicants using the German Credit Dataset. The project includes data exploration, model training, and an interactive web application built with Streamlit.

## Features

- **Data Analysis**: Comprehensive exploratory data analysis (EDA) with visualizations
- **Machine Learning Models**: Comparison of multiple algorithms:
  - Decision Tree Classifier
  - Random Forest Classifier
  - Extra Trees Classifier (Best Model)
  - XGBoost Classifier
- **Interactive Web App**: User-friendly Streamlit interface for real-time credit risk predictions
- **Model Persistence**: Trained models and encoders saved for easy deployment

## Project Structure

```
credit_risk/
│
├── main.py                          # Streamlit web application
├── credit.ipynb                     # Main data analysis and model training notebook
├── credit1.ipynb                    # Alternative analysis notebook
├── german_credit_data.csv           # Dataset
│
├── extra_tree_credit_model.pkl      # Trained Extra Trees model
├── Sex_encoder.pkl                  # Label encoder for Sex feature
├── Housing_encoder.pkl              # Label encoder for Housing feature
├── Saving accounts_encoder.pkl      # Label encoder for Saving accounts feature
├── Checking account_encoder.pkl     # Label encoder for Checking account feature
└── target_encoder.pkl               # Label encoder for target variable
```

## Dataset

The project uses the **German Credit Dataset** which contains information about 1000 credit applicants. The dataset includes:

- **Features**:
  - Age
  - Sex (male/female)
  - Job (0-3)
  - Housing (own/rent/free)
  - Saving accounts (little/moderate/rich/quite rich)
  - Checking account (little/moderate/rich)
  - Credit amount
  - Duration (in months)
  - Purpose

- **Target Variable**: Risk (Good/Bad)

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Steps

1. Clone or download this repository

2. Install required packages:

```bash
pip install streamlit pandas numpy scikit-learn xgboost joblib matplotlib seaborn
```

Or create a `requirements.txt` file with:

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
joblib>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

Then install:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Web Application

1. Make sure all model files (`.pkl` files) are in the project directory
2. Run the Streamlit app:

```bash
streamlit run main.py
```

3. The application will open in your default web browser at `http://localhost:8501`

4. Enter the applicant information:
   - Age (18-80)
   - Sex (male/female)
   - Job (0-3)
   - Housing (own/rent/free)
   - Saving Account (little/moderate/rich/quite rich)
   - Checking Account (little/moderate/rich)
   - Credit Amount
   - Duration (in months)

5. Click "Predict Risk" to get the prediction

### Model Training (Optional)

If you want to retrain the models:

1. Open `credit.ipynb` or `credit1.ipynb` in Jupyter Notebook
2. Run all cells to:
   - Load and explore the data
   - Preprocess the data (handle missing values, encode categorical variables)
   - Train and compare multiple models
   - Save the best model and encoders

## Model Performance

The models were evaluated using accuracy score with 5-fold cross-validation:

- **Decision Tree**: ~70.5% accuracy
- **Random Forest**: ~61.9% accuracy
- **Extra Trees**: ~64.8% accuracy (Selected as best model)
- **XGBoost**: ~61.9% accuracy

The **Extra Trees Classifier** was selected as the final model due to its best performance.

## Technologies Used

- **Python**: Programming language
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning library
- **XGBoost**: Gradient boosting framework
- **Matplotlib & Seaborn**: Data visualization
- **Joblib**: Model serialization

## Notes

- The model files (`.pkl`) must be present in the same directory as `main.py` for the application to work
- Missing values in the original dataset were handled by dropping rows with null values
- Categorical features are encoded using Label Encoders
- The model uses class weights to handle imbalanced data

## License

This project is for educational purposes.

## Author

Credit Risk Prediction Project

