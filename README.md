# 🎓 MBA Student Placement Prediction

## 📁 Repository Outline

This repository contains a comprehensive machine learning project focused on predicting MBA student job placement success. The project includes:

1. **P1M2_syarief_qayum.ipynb** - Main notebook containing the complete machine learning pipeline from data exploration to model deployment
1. **P1M2_syarief_qayum_inf.ipynb** - Inference notebook for model testing and validation
1. **deployment/** - Folder containing deployment files:
   - `streamlit_app.py` - Streamlit web application for interactive predictions
   - `prediction.py` - Prediction functions and utilities
   - `eda.py` - Exploratory data analysis functions
   - `model_xgb.pkl` - Trained XGBoost model
1. **Placement_Data_Full_Class.csv** - Dataset containing MBA student information
1. **README.md** - Project instructions and guidelines
1. **description.md** - This file - comprehensive project documentation

## 🏫 Problem Background

Universities with MBA programs face challenges in maximizing graduate job placement rates and effectively supporting students to excel in their careers. The main issue is the lack of data-driven understanding of how specific academic and non-academic factors influence placement success. Without these insights, it becomes difficult to:

- Identify tailored advice for students
- Optimize internal resource allocation
- Boost overall placement numbers

This project addresses these challenges by developing a predictive tool that assesses the likelihood of job placement for MBA students based on their academic performance, demographic information, and work experience.

## 🚀 Project Output

The project delivers:

1. **Predictive Model**: An XGBoost classification model that predicts whether an MBA student will secure job placement
1. **Web Application**: A Streamlit-based interactive application deployed on HuggingFace for real-time predictions
1. **Comprehensive Analysis**: Detailed exploratory data analysis revealing key factors influencing placement success
1. **Model Performance**: Baseline and hyperparameter-tuned models with detailed evaluation metrics

## 📊 Data

**Dataset**: Campus Recruitment Dataset from Kaggle

- **Source**: Kaggle - Factors Affecting Campus Placement
- **Size**: 215 students, 14 features
- **Features**:
  - Academic Performance: Secondary school percentage, higher secondary percentage, degree percentage, entrance test percentage, MBA percentage
  - Demographic Information: Gender, secondary school board, higher secondary board, specialization, degree type
  - Work Experience: Prior work experience (Yes/No)
  - Target Variables: Placement status (Placed/Not Placed), Salary (for placed students)

**Data Quality**: 
- No missing values except in salary column (expected for non-placed students)
- Well-balanced mix of categorical and numerical features
- Clean data requiring minimal preprocessing

## 🛠️ Method

The project follows a comprehensive machine learning pipeline:

### 1. **Exploratory Data Analysis (EDA)**

- Distribution analysis of categorical and numerical features
- Correlation analysis between features
- Target variable analysis (placement status distribution)
- Feature importance identification through visualization

### 2. **Feature Engineering**

- Data preprocessing and cleaning
- Handling missing values in salary column
- Feature encoding for categorical variables
- Data scaling and normalization
- Train-test split (80-20)

### 3. **Model Development**

- **Baseline Models**: Logistic Regression, Random Forest, Decision Tree, KNN, SVM, XGBoost
- **Model Selection**: Cross-validation with 5-fold CV
- **Hyperparameter Tuning**: GridSearchCV for XGBoost optimization
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

### 4. **Model Performance**

- **Best Model**: XGBoost (Baseline version)
- **Performance Metrics**:
  - Accuracy: 86%
  - Recall: 96%
  - Precision: 85%
  - F1-Score: 90%
  - ROC-AUC: 91%

## 💡 Key Findings

1. **Academic Performance**: Higher academic scores across all levels significantly improve placement chances
1. **Work Experience**: Students with prior work experience have better placement rates
1. **Degree Type**: Certain degree specializations show higher placement success
1. **Gender Distribution**: Slight gender imbalance in the dataset
1. **Model Performance**: XGBoost outperformed other algorithms in predicting placement success

## 🧰 Stacks

**Programming Language**: Python 3.11

**Core Libraries**:

- **Data Manipulation**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost
- **Visualization**: matplotlib, seaborn
- **Model Deployment**: streamlit, pickle
- **Data Balancing**: imbalanced-learn (SMOTE)
- **Feature Engineering**: feature-engine

**Tools & Platforms**:

- **Development**: Jupyter Notebook
- **Deployment**: Streamlit, HuggingFace
- **Version Control**: Git, GitHub

## ⚙️ Installation & Usage

### Prerequisites

```
pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlit
```

### Running the Application

1. Clone the repository
1. Navigate to the deployment folder
1. Run the Streamlit app:

```
streamlit run streamlit_app.py
```

### Using the Model

```
import pickle
import pandas as pd

# Load the model
with open('model_xgb.pkl', 'rb') as file:
    model = pickle.load(file)

# Make predictions
# Ensure input data matches the expected format
prediction = model.predict(input_data)
```

## 🌟 Business Impact

This predictive model provides several benefits for universities:

1. **Targeted Student Support**: Identify high-risk students early and provide targeted interventions
1. **Resource Optimization**: Allocate career services resources more effectively
1. **Curriculum Improvement**: Identify academic and non-academic factors that need enhancement
1. **Admission Strategy**: Use insights for better student selection and program design

## 🔮 Future Improvements

1. **Model Enhancement**: Further hyperparameter optimization with extended parameter grids
1. **Data Augmentation**: Collect additional features like student behavior, technical skills, company preferences
1. **Advanced Techniques**: Implement ensemble methods and advanced feature engineering
1. **Real-time Updates**: Continuous model retraining with new data

## 📚 Reference

- **Dataset**: Kaggle - Campus Recruitment
- **Deployment**: Streamlit application for interactive predictions
- **Documentation**: Complete analysis and insights in the main notebook

## ✍️ Author

**Syarief Qayum Suaib** - syarif.qayyum@gmail.com

- **Objective**: End-to-end machine learning project demonstrating data science skills from business understanding to model deployment
- **Focus**: Supervised learning classification
