# Lending-Club-Loan-Data-Analysis

📌 Project Overview

This project aims to predict whether a loan will be fully paid or defaulted using historical Lending Club loan data. By applying exploratory data analysis, feature engineering, and deep learning, the project demonstrates how machine learning can support risk-aware decision-making in the financial domain.

🎯 Objective

To build a deep learning model that accurately classifies loan applicants based on their likelihood of default, helping financial institutions minimize credit risk.

🗂️ Dataset Description

The dataset contains loan records from 2007 to 2015, including borrower financial details and loan characteristics.

🔑 Key Features

credit.policy – Credit underwriting compliance

purpose – Loan purpose

int.rate – Interest rate

installment – Monthly installment

log.annual.inc – Log of annual income

dti – Debt-to-income ratio

fico – FICO credit score

revol.bal – Revolving balance

revol.util – Revolving utilization

inq.last.6mths – Recent credit inquiries

delinq.2yrs – Past delinquencies

pub.rec – Public records

🎯 Target Variable

not.fully.paid

0 → Fully Paid

1 → Defaulted

🔍 Exploratory Data Analysis (EDA)

Analyzed the impact of FICO score, interest rate, and debt-to-income ratio on loan defaults

Identified patterns showing higher default risk for low credit scores and high interest rates

Visualized distributions and relationships using Matplotlib and Seaborn

🛠️ Feature Engineering

Converted categorical variables using One-Hot Encoding

Performed correlation analysis to remove redundant features

Applied feature scaling using StandardScaler

Prepared clean and optimized data for deep learning

🤖 Model Building

Implemented a Deep Neural Network (DNN) using Keras & TensorFlow

Used ReLU activation in hidden layers and Sigmoid activation in the output layer

Applied Dropout to reduce overfitting

⚙️ Model Details

Loss Function: Binary Crossentropy

Optimizer: Adam

Evaluation Metrics: Accuracy, Precision, Recall, F1-score

📈 Model Evaluation

Evaluated model performance using:

Confusion Matrix

Classification Report

The model successfully learned patterns from financial data to predict loan default risk.

🧠 Technologies Used

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

TensorFlow & Keras

Jupyter Notebook

✅ Results & Conclusion

The deep learning model effectively predicts loan default probability using borrower financial and credit data. This project highlights the importance of data preprocessing, feature engineering, and neural networks in solving real-world financial problems.

🔮 Future Improvements

Handle class imbalance using SMOTE or class weights

Try advanced models like Random Forest or XGBoost

Perform hyperparameter tuning

Deploy the model as a web application
