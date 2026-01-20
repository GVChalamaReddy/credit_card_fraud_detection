# credit_card_fraud_detection

## Project Overview
The objective of this project is to build a machine learning model that can accurately identify fraudulent credit card transactions. Fraudulent transactions are rare compared to legitimate ones, but they can cause significant financial losses. This project focuses on handling highly imbalanced datasets to ensure the model effectively captures the `minority class` (fraud) without generating excessive false alarms.

## Business Objective
For many banks, retaining high profitable customers is the number one business goal. Banking fraud, however, poses a significant threat to this goal for different banks. In terms of substantial financial losses, trust and credibility, this is a concerning issue to both banks and customers alike.In the banking industry, credit card fraud detection using machine learning is not only a trend but a necessity for them to put proactive monitoring and fraud prevention mechanisms in place. Machine learning is helping these institutions to reduce time-consuming manual reviews, costly chargebacks and fees as well as denials of legitimate transactions.This project is to **predict fraudulent credit card transactions with the help of machine learning models**.

## Data dictionary
The data set is taken from the **Kaggle website (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)** and has a total of **2,84,807 transactions**; out of these, **492 are fraudulent**. Since the data set is highly imbalanced, it needs to be handled before model building.The data set has also been modified with principal component analysis (PCA) to maintain confidentiality. Apart from `time` and `amount`, all the other features `(V1, V2, V3, up to V28)` are the principal components obtained using PCA. 

## Data Description
The dataset contains transactions made by credit cards in September 2013 by European cardholders.
- **Features**: It includes numerical features ($V1$ through $V28$) which are the result of a PCA (Principal Component Analysis) transformation to protect user privacy.
- **Time**: Contains the seconds elapsed between each transaction and the first transaction in the dataset.
- **Amount**: The transaction amount.
- **Class**: The target variable, where 1 indicates fraud and 0 indicates a legitimate transaction.
- **Imbalance**: Typically, fraud cases account for less than 0.2% of all transactions, making this a classic imbalanced classification problem.
  
## Methodology
The project follows a standard data science pipeline:
- **Data Preprocessing & EDA**:
  - **Skewness Mitigation**: Addressed heavy skewness in the Time and Amount variables using the Yeo-Johnson Power Transformation.
  - **Visualization**: Histograms and skewness coefficients were analyzed before and after transformation to ensure features were more normally distributed.
  - **Scaling**: Applied StandardScaler or PowerTransformer to ensure all features contribute equally, which is particularly critical for distance-based models like SVM and KNN.
  - **Splitting** data into Training and Testing sets using `Stratified Shuffling` to maintain class proportions.
- **Handling Imbalance**: To prevent the model from being biased toward the majority class, we implemented `ADASYN (Adaptive Synthetic Sampling)` or `SMOTE (Synthetic Minority Over-sampling Technique)` or `Random Over-sampling/Under-sampling` to balance the training data, within the training pipeline. This ensures that the model learns the distinct patterns of fraudulent behavior.
- **Feature Engineering**: Assessing if certain PCA components are more indicative of fraud.
- **Cross-Validation Strategy**
  - We employed a rigorous StratifiedKFold cross-validation approach with $k=[3, 5]$.
  - This strategy ensures that each fold maintains the same percentage of fraudulent transactions as the original dataset.

## Models Used
To ensure a robust comparison, the following machine learning models were implemented:
- **Logistic Regression**: Used as a baseline model for binary classification.
- **Random Forest Classifier**: A powerful ensemble method (Bagging) that reduces variance and handles non-linear patterns.
- **XGBoost Classifier**: An optimized Gradient Boosting (Boosting) framework designed for speed and high performance on structured data.

## Model Building / Training and Evaluation
The training and evaluation process was designed to address the class imbalance:
- **Training Strategy**: Models were trained on the balanced dataset (after oversampling/undersampling).
- **Cross-Validation**: We employed StratifiedKFold to ensure each fold remained representative of the minority class.
- **Hyperparameter Tuning**: We utilized GridSearchCV to optimize parameters such as learning rate, tree depth, and estimator counts.
- **Evaluation Priority**: In fraud detection, the cost of a `False Negative` (missing fraud) is much higher than a `False Positive`. Therefore, we prioritized `Recall` and `AUPRC` (Area Under the Precision-Recall Curve) over standard Accuracy.

## Evaluation Metrics
Given the extreme imbalance, Accuracy is not a reliable metric. We optimized and evaluated the models using:
- **AUC-ROC Score**: The primary metric used to find the best model, measuring the ability to distinguish between classes.
- **Precision-Recall Curves**: To understand the trade-off between capturing all fraud (Recall) and minimizing false alarms (Precision).
- **Confusion Matrix**: To visualize True Positives (caught fraud) and False Positives (customer friction).

## Model Performance and Results
The models were evaluated based on the following metrics:
- **Precision**: The accuracy of the positive predictions (avoiding false alarms).
- **Recall (Sensitivity)**: The ability to detect all actual fraud cases.
- **F1-Score**: The harmonic mean of Precision and Recall, providing a single score for balance.
- **ROC-AUC & AUPRC**: Measuring the trade-off between true positive and false positive rates.

`Note: The XGBoost model typically outperformed the baseline models, achieving a high AUPRC while maintaining a Recall above 80%.`

## Conclusion
This project demonstrates that while standard models achieve high accuracy on imbalanced data by simply predicting the majority class, they fail at the actual task of fraud detection. By applying `ADASYN` and utilizing ensemble methods like `XGBoost`, we significantly improved the model's ability to detect fraudulent patterns. The final model provides a reliable automated system for flagging suspicious transactions for further investigation.
  
## Technologies Used
- **python** - 3.13.1
- **numpy** - 2.2.1
- **pandas** - 2.2.3
- **matplotlib** - 3.10.0
- **seaborn** - 0.13.2
- **statsmodels** - 0.14.4
- **sklearn** - 1.6.1
  
## Acknowledgements

- This project was inspired by Upgrad IIIT Bangalore PG program on ML and AI.
- This project was part of Capstone Project.


## Contact
Created by @[GVChalamaReddy](https://github.com/GVChalamaReddy) - feel free to contact me!
