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
