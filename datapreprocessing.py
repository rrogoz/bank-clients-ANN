import pandas as pd
import numpy as np
import ownfunctions as f
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

# shuffle the data
shuffled_indices = np.arange(df.shape[0])
np.random.shuffle(shuffled_indices)
shuffled_inputs = inputs[shuffled_indices, :]
shuffled_targets = targets[shuffled_indices]

# split into validation and test

samples_count = shuffled_inputs.shape[0]
train_samples_count = int(0.8 * samples_count)
validation_samples_count = int(0.1 * samples_count)
test_count = samples_count - train_samples_count - validation_samples_count

train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]

validation_inputs = shuffled_inputs[train_samples_count:
                                    train_samples_count + validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:
                                      train_samples_count + validation_samples_count]

test_inputs = shuffled_inputs[train_samples_count +
                              validation_samples_count:]
test_targets = shuffled_targets[train_samples_count +
                                validation_samples_count:]


# save to
np.savez('Data_train', inputs=train_inputs, targets=train_targets)
np.savez('Data_validation',
         inputs=validation_inputs, targets=validation_targets)
np.savez('Data_test', inputs=test_inputs, targets=test_targets)
