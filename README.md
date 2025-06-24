# Graduate Salary Prediction using PySpark and ML Models

This project aims to predict the salaries of engineering graduates using various machine learning models in **PySpark**, such as Linear Regression, Decision Trees, Random Forests, and Gradient Boosting. The analysis is performed in **Google Colab** and uses **PySpark**, **Matplotlib**, **Seaborn**, and other Python tools for EDA and modeling.

## 📁 Dataset

- **File:** `Engineering_graduate_salary.csv`
- Contains academic, personality, and aptitude test data for engineering students.

## 🔍 Features Used

- Academic scores: `10percentage`, `12percentage`, `collegeGPA`
- Aptitude tests: `English`, `Logical`, `Quant`, `Domain`
- Personality traits: `conscientiousness`, `agreeableness`, `extraversion`, `nueroticism`, `openess_to_experience`
- Target variable: `Salary`

## 🧪 Models Implemented

| Model                  | RMSE     | R² Score | Adjusted R² | MAE     |
|------------------------|----------|----------|--------------|---------|
| Linear Regression      | 207,610  | ~0.06    | ~0.05        | ~       |
| Decision Tree Regressor| 277,366  | -0.21    | -            | 126,496 |
| Random Forest Regressor| 246,284  | 0.045    | -            | 116,799 |
| Gradient Boosted Trees | 276,148  | -0.20    | -            | 122,840 |

## 📊 Visualizations

- Correlation bar plot with Salary
- Salary histogram and boxplot
- Residual plots for all models
- Actual vs Predicted scatter plots

## 🚀 How to Run

1. Open the notebook in [Google Colab](https://colab.research.google.com/).
2. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
Upload Engineering_graduate_salary.csv to your drive and update the file path in:

df = spark.read.csv('/content/drive/My Drive/Engineering_graduate_salary.csv', header=True, inferSchema=True)

Run all cells for data preprocessing, EDA, feature engineering, model training, and evaluation.
