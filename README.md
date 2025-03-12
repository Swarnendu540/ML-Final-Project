# Movie Rating Prediction Project

## 📌 Introduction
### 📖 Background
Movie ratings play a crucial role in the entertainment industry by influencing audience preferences, revenue generation, and critical acclaim. Accurately predicting a movie's rating can benefit producers, distributors, and viewers by providing insights into a film's potential success. This project leverages machine learning techniques to predict IMDb movie ratings based on various features such as **Metascore, Votes, and Gross revenue**.

### 🎯 Motivation
The motivation behind this project is to demonstrate practical applications of **machine learning** in the film industry. By analyzing and predicting movie ratings, stakeholders can make informed decisions regarding **movie marketing, investments, and audience engagement strategies**. Additionally, this project explores multiple machine learning techniques, enhancing the understanding of regression-based predictive modeling.

### 🎯 Objectives
The main objectives of this project are:
- **Comprehensive preprocessing** of movie-related data.
- **Predicting key variables**, such as Metascore, using advanced machine learning techniques.
- **Comparative analysis** of different predictive models to determine the most effective approach.

## 📊 Dataset Description
- **Data Source**: IMDb and Kaggle datasets ([Link](https://www.kaggle.com/datasets/prishasawhney/imdb-dataset-top-2000-movies))
- **Key Features**:
  - **Metascore**: Critical reception metric
  - **Votes**: Audience engagement indicator
  - **Gross**: Financial performance marker
- **Target Variable**: IMDb rating

## 🛠 Data Preprocessing
- **Handling Missing Values**:
  - **K-Nearest Neighbors (KNN) imputation** for data completeness.
- **Feature Engineering**:
  - Conversion of non-numeric columns to numeric formats.
  - Removal of non-numeric symbols (e.g., commas from Votes).
  - Feature selection using **statistical correlation analysis**.
- **Outlier Detection & Handling**
- **Data Standardization & Categorical Encoding**

## 🤖 Machine Learning Models Used
- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**
- **Elastic Net**
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **XGBoost Regressor**
- **Support Vector Regression (SVR)**
- **MLP Regressor**

## 📈 Model Evaluation
- **Performance Metrics**:
  - **R-squared (R²) Coefficient**
  - **Root Mean Squared Error (RMSE)**
  - **Mean Absolute Error (MAE)**
- **Best Performing Model**: **XGBoost**
  - RMSE: **0.31**
  - R² Score: **0.67**
  - Strongest generalization and feature handling capabilities.

## 🏗 Implementation Details
- **Python Libraries Used**:
  - **Data Manipulation**: `pandas`, `NumPy`
  - **Machine Learning**: `scikit-learn`, `XGBoost`
  - **Visualization**: `matplotlib`, `seaborn`
- **Hyperparameter Optimization**:
  - `GridSearchCV` for fine-tuning models.
  - Optimal tuning for SVR (`C`, `gamma`, `epsilon`) & Random Forest (`n_estimators`, `max_depth`).
- **Training Strategy**:
  - Train-Test Split: **70%-15%-15%**
  - **Cross-validation for generalization**

## 🔥 Key Insights & Findings
- **Ensemble models (Random Forest, Gradient Boosting, XGBoost) outperform traditional regression models.**
- **Feature engineering significantly impacts model performance.**
- **Regularization (Ridge/Lasso) reduces overfitting while maintaining accuracy.**

## 🚀 Future Improvements
- **Explore Deep Learning models for enhanced prediction accuracy.**
- **Incorporate more features (e.g., Director, Genre, Actor influence).**
- **Improve data augmentation techniques to handle missing data better.**

## 📜 Authors
**Team Swarvik**

## 📢 Contributing
Feel free to fork, raise issues, and contribute to this project!

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

