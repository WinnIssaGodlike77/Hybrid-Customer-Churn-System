# Hybrid Customer Churn Prediction

## Project Overview
This project predicts customer churn using a hybrid machine learning approach that combines customer segmentation with supervised churn prediction.

## Business Problem
Customer churn reduces revenue and increases customer acquisition costs. This project helps identify high-risk customers and provides retention actions.

## Dataset
Telco Customer Churn dataset from Kaggle.

## Methods Used
- Data Cleaning
- Exploratory Data Analysis
- KMeans Clustering
- Random Forest Classification
- Logistic Regression Benchmark
- Risk Scoring
- Retention Recommendation

## Tools and Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Project Outputs
- customer_churn_with_clusters.csv
- churn_prediction_results.csv
- model_comparison.csv
- feature_importance.csv
- rf_churn_model.pkl

## Key Findings
- Short-tenure customers are more likely to churn
- Month-to-month customers have higher churn
- Certain clusters show significantly higher churn rates
- Random Forest outperformed Logistic Regression

## Business Recommendations
- Retain high-risk customers with targeted offers
- Focus on short-tenure and high-risk clusters
- Improve service engagement for medium-risk customers

## How to Run
1. Install the requirements
2. Open the notebook
3. Run all cells in order




# Hybrid Customer Churn Prediction and Segmentation System

## Project Overview

Customer churn is a critical challenge for subscription-based businesses such as telecommunications, banking, and online services. Losing customers directly impacts revenue and long-term growth. Therefore, organizations aim to identify customers who are likely to churn and take preventive actions before they leave.

This project develops a **hybrid customer churn prediction system** that combines:

* **Customer Segmentation (Unsupervised Learning)**
* **Churn Prediction (Supervised Machine Learning)**

The goal is to provide **data-driven insights** that help businesses identify high-risk customers and apply targeted retention strategies.

The system uses clustering techniques to segment customers and then applies machine learning models to predict churn probabilities.

---

## Objectives

The main objectives of this project are:

* Analyze customer behavior patterns
* Segment customers into meaningful groups
* Predict which customers are likely to churn
* Provide churn risk levels
* Suggest retention strategies for high-risk customers
* Build a professional end-to-end data science pipeline

---

## Dataset

This project uses the **Telco Customer Churn dataset** from Kaggle.

The dataset contains customer information such as:

* Customer tenure
* Monthly charges
* Total charges
* Contract type
* Internet service
* Payment method
* Demographic information
* Churn status

Each record represents a unique customer.

---

## Project Workflow

The project follows a structured data science pipeline:

1. Data Loading
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Customer Segmentation using **K-Means Clustering**
6. Cluster Validation using **Silhouette Score**
7. Machine Learning Model Training
8. Model Comparison and Evaluation
9. Business Interpretation of Results
10. Exporting Outputs and Model Artifacts

---

## Technologies and Tools

The following tools and libraries were used in this project:

Programming Language

* Python

Data Processing

* Pandas
* NumPy

Visualization

* Matplotlib
* Seaborn

Machine Learning

* Scikit-learn

Development Environment

* Jupyter Notebook
* VS Code

Version Control

* Git
* GitHub

---

## Machine Learning Models Used

Two machine learning models were implemented:

### Random Forest Classifier

Used as the primary churn prediction model because it handles complex feature interactions and nonlinear relationships effectively.

Key features:

* Handles class imbalance using `class_weight="balanced"`
* Robust performance for tabular datasets

### Logistic Regression

Used as a benchmark model for comparison.

Advantages:

* Simple and interpretable
* Useful baseline for classification problems

---

## Customer Segmentation

Customer segmentation was performed using **K-Means clustering**.

Features used for clustering:

* Tenure
* Monthly Charges
* Total Charges

Cluster quality was evaluated using:

* **Elbow Method**
* **Silhouette Score**

The clustering step helps identify different types of customers such as:

* Long-term loyal customers
* High-spending customers
* New customers
* High-risk customers

---

## Model Evaluation Metrics

The models were evaluated using several performance metrics:

* Accuracy
* ROC-AUC Score
* Confusion Matrix
* Classification Report

These metrics provide insights into the predictive performance of the churn prediction models.

---

## Visualizations Generated

The project produces multiple visual outputs including:

* Churn distribution chart
* Customer tenure vs churn
* Monthly charges vs churn
* Contract type vs churn
* Elbow method plot
* Customer cluster scatter plot
* Churn rate by cluster
* Confusion matrix heatmap
* ROC curve
* Feature importance chart

All visualizations are saved in the **visuals/** directory.

---

## Business Insights

The model outputs are transformed into business-friendly results:

### Churn Risk Levels

Customers are categorized into three risk groups:

Low Risk
Medium Risk
High Risk

### Retention Recommendations

The system also generates recommended actions:

High Risk
→ Offer discounts or direct retention calls

Medium Risk
→ Send engagement emails or loyalty rewards

Low Risk
→ Maintain regular service

This allows businesses to apply targeted retention strategies.

---

## Project Outputs

The project automatically generates the following files:

customer_churn_with_clusters.csv
Customer dataset with cluster labels

churn_prediction_results.csv
Prediction results with churn probabilities

model_comparison.csv
Performance comparison of models

feature_importance.csv
Important features influencing churn

rf_churn_model.pkl
Saved trained machine learning model

visuals/
Folder containing all generated charts

---

## Project Structure

```
Customer-Churn-Prediction
│
├── churn_hybrid_analysis.ipynb
├── customer_churn.csv
├── customer_churn_with_clusters.csv
├── churn_prediction_results.csv
├── model_comparison.csv
├── feature_importance.csv
├── rf_churn_model.pkl
│
├── requirements.txt
├── README.md
│
└── visuals
    ├── churn_distribution.png
    ├── elbow_method.png
    ├── cluster_scatterplot.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── feature_importance.png
```

---

## How to Run the Project

1. Clone the repository

```
git clone https://github.com/yourusername/customer-churn-prediction.git
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Run the notebook

Open the notebook in Jupyter or VS Code and execute the cells sequentially.

---

## Future Improvements

Possible future improvements include:

* Hyperparameter tuning using GridSearchCV
* Deploying the model with Flask or FastAPI
* Building an interactive dashboard using Power BI or Streamlit
* Adding SHAP explainability analysis
* Automating retraining pipelines

---

## Author

Win Phyo Thein Han 
Computer Systems Engineering (Computer Science) Graduate – University of Sunderland
Interested in Data Analytics, Data Science, and AI/ML Engineering.

---
