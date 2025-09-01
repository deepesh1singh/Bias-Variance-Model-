# Customer Churn Prediction using XGBoost

## Project Overview

This project implements an XGBoost classifier to predict customer churn for a fictional telecommunications company. The goal is to identify customers who are likely to leave the company based on various customer attributes and service usage patterns.

## Problem Statement

The Telco customer churn dataset contains information about a fictional telco company that provided home phone and Internet services to 7,043 customers in California in Q3. The objective is to implement an XGBoost classifier to predict whether a customer will leave or stay with the company.

## Dataset Description

The dataset contains 33 features including:

### Customer Demographics
- **CustomerID**: Unique identifier for each customer
- **Gender**: Male/Female
- **Senior Citizen**: Yes/No (65 or older)
- **Partner**: Yes/No (has a partner)
- **Dependents**: Yes/No (lives with dependents)

### Location Information
- **Country**: Customer's primary residence country
- **State**: Customer's primary residence state
- **City**: Customer's primary residence city
- **Zip Code**: Customer's zip code
- **Latitude/Longitude**: Geographic coordinates

### Service Information
- **Tenure Months**: Total months with the company
- **Phone Service**: Yes/No (home phone service)
- **Multiple Lines**: Yes/No (multiple telephone lines)
- **Internet Service**: No/DSL/Fiber Optic/Cable
- **Online Security**: Yes/No (additional security service)
- **Online Backup**: Yes/No (additional backup service)
- **Device Protection**: Yes/No (equipment protection plan)
- **Tech Support**: Yes/No (technical support plan)
- **Streaming TV**: Yes/No (streaming television)
- **Streaming Movies**: Yes/No (streaming movies)

### Contract & Billing
- **Contract**: Month-to-Month/One Year/Two Year
- **Paperless Billing**: Yes/No
- **Payment Method**: Bank Withdrawal/Credit Card/Mailed Check
- **Monthly Charges**: Current monthly charge for all services
- **Total Charges**: Total charges to end of quarter

### Target Variables
- **Churn Value**: Binary target (1 = left, 0 = remained)
- **Churn Label**: Yes/No (customer left/remained)
- **Churn Score**: Predictive score (0-100) from IBM SPSS Modeler
- **CLTV**: Customer Lifetime Value
- **Churn Reason**: Specific reason for leaving

## Technical Implementation

### Data Preprocessing

1. **Data Cleaning**
   - Removed columns with exit interview information (Churn Label, Churn Score, CLTV, Churn Reason)
   - Eliminated columns with single unique values
   - Removed CustomerID and Lat Long (redundant with separate coordinates)
   - Replaced whitespaces with underscores in column names and values

2. **Missing Value Handling**
   - Identified 11 missing values in Total_Charges column
   - Filled missing values with 0 (customers with 0 tenure months)
   - Converted Total_Charges to float64 datatype

3. **Feature Engineering**
   - Applied one-hot encoding to categorical variables
   - Expanded feature set from 23 to 1,178 dimensions
   - Maintained numerical features unchanged

### Model Architecture

**Algorithm**: XGBoost (eXtreme Gradient Boosting)
- **Objective Function**: binary:logistic
- **Evaluation Metric**: AUC (Area Under Curve)
- **Random State**: 42 (for reproducibility)

**Key Hyperparameters**:
- max_depth: Maximum tree depth
- learning_rate: Step size shrinkage
- gamma: Minimum loss reduction for partition
- reg_lambda: L2 regularization parameter
- scale_pos_weight: Positive class weight scaling
- colsample_bytree: Column subsample ratio
- subsample: Training instance subsample ratio

### Data Splitting Strategy

- **Train-Validation Split**: 80-20 split with stratification
- **Stratification**: Maintains class proportions (26.54% churn rate)
- **Random State**: 42 for reproducible results

## Performance Metrics

### Model Performance
- **Training AUC**: 0.9395
- **Validation AUC**: 0.8500
- **Overall Accuracy**: 80%
- **Class Distribution**: 26.54% churn rate

### Classification Report
```
              precision    recall  f1-score   support

           0       0.85      0.89      0.87      1035
           1       0.64      0.55      0.59       374

    accuracy                           0.80      1409
   macro avg       0.74      0.72      0.73      1409
weighted avg       0.79      0.80      0.79      1409
```

### Key Insights
- **Class Imbalance**: Significant imbalance between churned (26.54%) and retained customers (73.46%)
- **Model Performance**: Good performance on majority class (0), moderate performance on minority class (1)
- **Overfitting**: Training AUC (0.9395) higher than validation AUC (0.8500) indicates some overfitting

## Hyperparameter Tuning

The project implements hyperparameter optimization using RandomizedSearchCV:

**Search Space**:
- max_depth: [3, 4, 5, 6, 7, 8, 9, 10]
- learning_rate: [0.01, 0.1, 0.3]
- n_estimators: [100, 200, 300]

**Best Parameters**:
- n_estimators: 100
- max_depth: 3
- learning_rate: 0.01

## Feature Importance Analysis

The model provides multiple importance metrics:
- **Weight**: Number of times a feature is used in branches
- **Gain**: Average gain across splits
- **Cover**: Average coverage across splits
- **Total Gain**: Total gain across all splits
- **Total Cover**: Total coverage across all splits

## Visualization Features

1. **ROC Curves**: Training vs. validation performance
2. **Confusion Matrix**: Binary classification results
3. **AUC-ROC Plots**: Model discrimination ability
4. **Tree Visualization**: Optional XGBoost tree structure

## Installation Requirements

### Core Dependencies
```bash
pandas
numpy
xgboost
scikit-learn
matplotlib
```

### Optional Dependencies
```bash
conda install graphviz python-graphviz
```

### Python Environment
- Python 3.x
- Jupyter Notebook support
- Random seed set to 42 for reproducibility

## Usage Instructions

1. **Data Loading**: Ensure the dataset file `Telco_customer_churn.xlsx` is in the working directory
2. **Notebook Execution**: Run all cells sequentially in the Jupyter notebook
3. **Model Training**: The notebook automatically handles data preprocessing and model training
4. **Results Analysis**: View performance metrics, visualizations, and classification reports

## File Structure

```
project/
├── 22CS30020_P2.ipynb          # Main Jupyter notebook
├── README.md                    # This documentation file
└── Telco_customer_churn.xlsx   # Dataset file (not included)
```

## Methodology

### 1. Exploratory Data Analysis
- Dataset overview and feature understanding
- Missing value identification and treatment
- Data type conversion and validation

### 2. Feature Engineering
- Categorical variable encoding
- Whitespace normalization
- Feature selection and dimensionality reduction

### 3. Model Development
- XGBoost classifier implementation
- Hyperparameter optimization
- Cross-validation and early stopping

### 4. Performance Evaluation
- Multiple evaluation metrics
- Visualization of results
- Model interpretation and feature importance

## Limitations and Considerations

1. **Class Imbalance**: 26.54% churn rate requires careful handling
2. **Feature Engineering**: High-dimensional feature space (1,178 features)
3. **Overfitting**: Training performance significantly better than validation
4. **Data Quality**: Missing values and data type inconsistencies

## References

- **Dataset Source**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset)
- **IBM Documentation**: [Telco Customer Churn Analysis](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)
- **XGBoost Documentation**: [Official XGBoost Guide](https://xgboost.readthedocs.io/)

---

*This project demonstrates the implementation of XGBoost for customer churn prediction, showcasing data preprocessing, model development, and performance evaluation in a real-world business context.*

