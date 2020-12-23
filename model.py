import numpy as np
import pandas as pd

import sklearn
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics
from sklearn import ensemble

# Resampling
from imblearn import over_sampling, under_sampling

# Model Saving
import pickle

# Loading Data
df = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv" )

# Dropping Features
drop_features = ['Loan_ID', 'Unnamed: 0']
df = df.drop(drop_features, axis=1)

# Filling with Mean and Mode values
numerical_missing_features = ['LoanAmount', 'Loan_Amount_Term']
cat_missing_features = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']

for i in numerical_missing_features:
    mean_value = int(df[i].mean())
    print(str(i) + '-' + str(mean_value))
    df[i] = df[i].fillna(mean_value)

for i in cat_missing_features:
    mode_value = df[i].mode().to_list()[0]
    print(str(i) + '-' + str(mode_value))
    df[i] = df[i].fillna(mode_value)



# Encoding
le_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
label_encoder = preprocessing.LabelEncoder()

for i in le_features:
    df[i] = label_encoder.fit_transform(df[i])


# Splitting Data
target = 'Loan_Status'
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Oversampling
ros = over_sampling.SMOTE(random_state=42,  sampling_strategy = 1.0)
x_ros, y_ros = ros.fit_resample(X_train, y_train)

# Model Building
forest = ensemble.RandomForestClassifier(random_state=3)
forest.fit(x_ros, y_ros)


# Inference
y_pred = forest.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
print(metrics.f1_score(y_test, y_pred))


# Serialization of Model
filename = 'rfmodel.pkl'
with open(filename, 'wb') as file:
    pickle.dump(forest, file)


# Deserialization of Model
with open(filename, 'rb') as file:
    forest = pickle.load(file)
