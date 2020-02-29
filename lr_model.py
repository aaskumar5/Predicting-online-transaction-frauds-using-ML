# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Import Data

data = pd.read_csv('D:/capstone project/Predicting-online-transaction-frauds-using-ML/creditcardcsvpresent.csv')
data.drop('Transaction date',axis = 1, inplace = True)
data = data.dropna()

# Data manipulation

data['Is declined']=data['Is declined'].replace({'N': 0, 'Y': 1})
data['isForeignTransaction']=data['isForeignTransaction'].replace({'N': 0, 'Y': 1})
data['isHighRiskCountry']=data['isHighRiskCountry'].replace({'N': 0, 'Y': 1})
data['isFradulent']=data['isFradulent'].replace({'N': 0, 'Y': 1})

# Splitting the predictors and the target

x = data.iloc[:,1:10].values
y = data.iloc[:,10:11].values

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# Logistic Regression Model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train.ravel())

Y_pred = classifier.predict(X_test)

# Saving model to disk


pickle.dump(classifier, open('model.pkl','wb'))

# Confusion Matrix

cm = confusion_matrix(Y_test, Y_pred)
print(cm)

