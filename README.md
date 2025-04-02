Fraud Detection Model

Overview

This project aims to develop a robust fraud detection system to classify fraudulent transactions. By using machine learning models such as Logistic Regression, Random Forest Classifier, and Gradient Boosting Classifier, the goal is to predict and identify fraudulent accounts and transactions in the Indicina dataset. We also apply techniques like SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance and enhance model performance.

Table of Contents

Project Description

1. Data

2. Technologies Used

3. Model Building

4. Evaluation Metrics

5. Installation

6. Assumptions

7. License

Project Description

This project is designed to help identify fraudulent accounts by training machine learning models on a dataset of customer transactions. Fraud detection is a critical task in financial institutions, and this model aims to detect fraudulent activity based on customer data such as account balances, transaction history, and other features.

Goal: Detect fraud with high accuracy while ensuring that the model can identify fraud cases in an imbalanced dataset.

Approach: Various classifiers are trained, including Logistic Regression, Random Forest Classifier, and Gradient Boosting Classifier. We use SMOTE to balance the dataset and perform Recursive Feature Elimination (RFE) to select the most important features.

Data

The dataset used in this project contains customer transaction information. The key features include:

1. cust_id: Customer ID

2. activated_year: Year the account was activated

3. last_payment_year: Year the last payment was made

4. cash_advance: Amount of cash advanced by the customer

5. credit_limit: Credit limit assigned to the customer

6. balance: Customer's balance in their account

7. fraud_label: The target variable (1 for fraud, 0 for non-fraud)

Data was preprocessed to ensure clean, usable data, with transformations such as:

1. Removing unwanted characters from IDs

2. Normalizing date formats

3. Calculating the percentage of cash advances

Technologies Used

1. Python: Core language for the project

2. pandas: Data manipulation and analysis

3. NumPy: Numerical operations

4. scikit-learn: Machine learning models and preprocessing

5. imblearn: Library for handling imbalanced datasets (SMOTE)

6. matplotlib and seaborn: Data visualization

7. Jupyter Notebooks: Interactive development environment

Model Building

Step 1: Data Preprocessing

1. Scaling: StandardScaler is used to scale the feature data to have a mean of 0 and a standard deviation of 1.

2. Feature Selection: Recursive Feature Elimination (RFE) is used to select the top features based on their importance to the model.

3. SMOTE: SMOTE is applied to balance the dataset by generating synthetic samples of the minority class (fraudulent accounts).

Step 2: Model Training

The following models are used:

1. Logistic Regression: A simple and interpretable model for binary classification.

2. Random Forest Classifier: An ensemble learning model that builds multiple decision trees and combines their results.

3. Gradient Boosting Classifier: A boosting method that combines weak learners to create a strong model.

Each model is trained on the preprocessed data and evaluated on a test set using performance metrics.

Step 3: Evaluation Metrics

The models are evaluated using the following metrics:

1. Accuracy: Measures the overall correctness of the model.

2. ROC-AUC Score: Measures the model’s ability to distinguish between fraud and non-fraud cases.

3. Classification Report: Provides precision, recall, and F1-score for both fraud and non-fraud classes.

Evaluation Metrics

After training the models, they were evaluated based on the following criteria:

1. Precision: The ability of the model to correctly identify fraud cases (minimizing false positives).

2. Recall: The ability of the model to correctly identify fraudulent accounts (minimizing false negatives).

3. ROC-AUC: Measures the overall ability of the model to distinguish between fraudulent and non-fraudulent accounts.

Installation

Prerequisites

Python 3.x

pip (Python package installer)

Installing Required Libraries

To run this project, you need to install the following Python libraries:

pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn

Running the Project

Clone the repository or download the project files.

Place the dataset in the appropriate folder.

Run the Jupyter Notebook or Python script to begin the analysis.

Assumptions

The dataset is assumed to be complete and pre-processed, with no missing values.

The models and techniques used (Logistic Regression, Random Forest, Gradient Boosting, SMOTE) are appropriate for the fraud detection problem and data at hand.

Class imbalance is a significant issue, so SMOTE is applied to help balance the dataset and improve model performance.

The models assume that the features used for training the classifiers are relevant and provide meaningful information about fraud detection.

License

This project is licensed under the MIT License - see the LICENSE file for details.

