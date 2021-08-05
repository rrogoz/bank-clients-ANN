import pandas as pd
import numpy as np
import ownfunctions as f
from sklearn.model_selection import train_test_split
# data reading
rawData = pd.read_csv('Churn_Modelling.csv')
# removing unnecessary columns
colsToPopAtStart = ['RowNumber', 'CustomerId', 'Surname']
df = rawData.drop(colsToPopAtStart, axis=1)

# Mapping Values
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

# One-hot encoding of geography columns
colsToOneHotDummy = ['Geography']
df = f.one_hot_dummies(df, colsToOneHotDummy, True)

# standardizing the data
colsToStandardize = ['CreditScore', 'Age', 'Tenure', 'Balance',
                     'NumOfProducts', 'EstimatedSalary']
df = f.standardize(df, colsToStandardize)


inputs = df.iloc[:, 0:-1].to_numpy()
targets = df.iloc[:, -1].to_numpy()

xTrain, xTest = train_test_split(inputs, test_size=0.2, random_state=0)
yTrain, yTest = train_test_split(targets, test_size=0.2, random_state=0)
