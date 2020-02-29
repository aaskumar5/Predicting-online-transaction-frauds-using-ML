# Importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

# Import Data

data = pd.read_csv('C:/Users/Magilan/Desktop/Sem 7/FYP/proj/creditcardcsvpresent.csv')
data.drop('Transaction date',axis = 1, inplace = True)


data['Is declined']=data['Is declined'].replace({'N': 0, 'Y': 1})
data['isForeignTransaction']=data['isForeignTransaction'].replace({'N': 0, 'Y': 1})
data['isHighRiskCountry']=data['isHighRiskCountry'].replace({'N': 0, 'Y': 1})
data['isFradulent']=data['isFradulent'].replace({'N': 0, 'Y': 1})

# Splitting the predictors and the target

x = data.iloc[:,1:10].values
y = data.iloc[:,10:11].values

X = data.iloc[:, 1:10].columns
Y = data.iloc[:1, 10: ].columns
print(X)
print(Y)

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
print("Length of X_train is: {X_train}".format(X_train = len(X_train)))
print("Length of X_test is: {X_test}".format(X_test = len(X_test)))
print("Length of y_train is: {Y_train}".format(Y_train = len(Y_train)))
print("Length of y_test is: {Y_test}".format(Y_test = len(Y_test)))

# Preliminary analysis

# distribution of anomalous features

features = data.iloc[:,1:10].columns


plt.figure(figsize=(12,28*4))
gs = gridspec.GridSpec(28, 1)

for i, c in enumerate(data[features]):
 ax = plt.subplot(gs[i])
 sns.distplot(data[c][data.isFradulent == 1], bins=50)
 sns.distplot(data[c][data.isFradulent == 0], bins=50)
 ax.set_xlabel('')
 ax.set_title('histogram of feature: ' + str(c))
plt.show()

# Determine number of fraud cases in dataset

Fraud = data[data['isFradulent'] == 1]
Valid = data[data['isFradulent'] == 0]

outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)

print('Fraud Cases: {}'.format(len(data[data['isFradulent'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['isFradulent'] == 0])))

print('Amount details of fraudulent transaction')
Fraud.Transaction_amount.describe()

print('Amount details of legitimate transaction')
Valid.Transaction_amount.describe()

# Correlation matrix

corrmat = data.corr()
fig = plt.figure(figsize = (8, 5))
sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()

# Standardising

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

# Logistic Regression Model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

cm = confusion_matrix(Y_test, Y_pred)
print(cm)

# Evaluation

from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_recall_fscore_support,precision_score

accuracy_lr = accuracy_score(Y_test, Y_pred)
print("Accuracy of the Logistic Regrssion Model ",accuracy_lr)
f1_score_lr = f1_score(Y_test, Y_pred)
recall_score_lr = recall_score(Y_test, Y_pred)
precision_score_lr = precision_score(Y_test, Y_pred)
print("Sensitivity/Recall for Logistic Regression Model 1 : {recall_score}".format(recall_score = recall_score_lr))
print("Precision of the Logistic Regrssion Model ",precision_score_lr)
print("F1 Score for Logistic Regression Model 1 : {f1_score}".format(f1_score = f1_score_lr))

precision_recall_fscore_support(Y_test, Y_pred)

# KNN Model

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 20, metric = 'minkowski', p = 2)
classifier.fit(X_train_sc, Y_train)

y_pred = classifier.predict(X_test_sc)
print(Y_pred)

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(Y_test, Y_pred)
print(cm1)

# Evaluation

from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_recall_fscore_support,precision_score

accuracy_knn = accuracy_score(Y_test, Y_pred)
print("Accuracy of the KNN Model ",accuracy_knn)
f1_score_knn = f1_score(Y_test, Y_pred)
recall_score_knn = recall_score(Y_test, Y_pred)
precision_score_knn = precision_score(Y_test, Y_pred)
print("Sensitivity/Recall for KNN Model : {recall_score}".format(recall_score = recall_score_knn))
print("Precision of the KNN Model ",precision_score_knn)
print("F1 Score for KNN Model : {f1_score}".format(f1_score = f1_score_knn))


# Descision Tree 

from sklearn.tree import DecisionTreeClassifier 
dtree_model = DecisionTreeClassifier(max_depth = 1).fit(X_train, Y_train) 
dtree_predictions = dtree_model.predict(X_test) 
  

cm2 = confusion_matrix(Y_test, dtree_predictions) 
print(cm2)

# Evaluation

from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_recall_fscore_support,precision_score

accuracy_dtree = accuracy_score(Y_test, dtree_predictions)
print("Accuracy of the Decision Tree Model ",accuracy_dtree)
f1_score_dtree = f1_score(Y_test, dtree_predictions)
recall_score_dtree = recall_score(Y_test, dtree_predictions)
precision_score_dtree = precision_score(Y_test, dtree_predictions)
print("Sensitivity/Recall for Decision Tree Model : {recall_score}".format(recall_score = recall_score_dtree))
print("Precision of the Decision Tree Model ",precision_score_dtree)
print("F1 Score for Decision Tree Model : {f1_score}".format(f1_score = f1_score_dtree))

# SVM

from sklearn.svm import SVC
classifier_SVM = SVC(kernel = 'linear', random_state = 0)
classifier_SVM.fit(X_train_sc, Y_train)
svm_pred = classifier_SVM.predict(X_test_sc)

cm3 = confusion_matrix(Y_test, svm_pred)
print(cm3)

# Evaluation

from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_recall_fscore_support,precision_score

accuracy_svm = accuracy_score(Y_test, svm_pred)
print("Accuracy of the SVM Model ",accuracy_svm)
f1_score_svm = f1_score(Y_test, svm_pred)
recall_score_svm = recall_score(Y_test, svm_pred)
precision_score_svm = precision_score(Y_test, svm_pred)
print("Sensitivity/Recall for SVM Model : {recall_score}".format(recall_score = recall_score_svm))
print("Precision of the SVM Model ",precision_score_svm)
print("F1 Score for SVM Model : {f1_score}".format(f1_score = f1_score_svm))

# Naive Bayes

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train_sc, Y_train)

y_pred = classifier.predict(X_test_sc)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
print(cm)

# Evaluation

from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_recall_fscore_support,precision_score

accuracy_naive = accuracy_score(Y_test, y_pred)
print("Accuracy of the Naive Bayes Model ",accuracy_naive)
f1_score_naive = f1_score(Y_test, y_pred)
recall_score_naive = recall_score(Y_test, y_pred)
precision_score_naive = precision_score(Y_test, y_pred)
print("Sensitivity/Recall for Naive Bayes Model : {recall_score}".format(recall_score = recall_score_naive))
print("Precision of the Naive Bayes Model ",precision_score_naive)
print("F1 Score for Naive Bayes Model : {f1_score}".format(f1_score = f1_score_naive))

# Random Forest

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)

cm4 = confusion_matrix(Y_test, y_pred)
print(cm4)

# Evaluation

from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_recall_fscore_support,precision_score

accuracy_rf = accuracy_score(Y_test, y_pred)
print("Accuracy of the Naive Bayes Model ",accuracy_rf)
f1_score_rf = f1_score(Y_test, y_pred)
recall_score_rf = recall_score(Y_test, y_pred)
precision_score_rf = precision_score(Y_test, y_pred)
print("Sensitivity/Recall for Random Forest Model : {recall_score}".format(recall_score = recall_score_rf))
print("Precision of the Random Forest Model ",precision_score_rf)
print("F1 Score for Random Forests Model : {f1_score}".format(f1_score = f1_score_rf))

# XGBoost

from sklearn.ensemble import GradientBoostingClassifier

modelXG= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
modelXG.fit(X_train, Y_train)
y_pred= modelXG.predict(X_test)

cm5 = confusion_matrix(Y_test, y_pred)
print(cm5)

# Evaluation

from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_recall_fscore_support,precision_score

accuracy_xg = accuracy_score(Y_test, y_pred)
print("Accuracy of the Naive Bayes Model ",accuracy_xg)
f1_score_xg = f1_score(Y_test, y_pred)
recall_score_xg = recall_score(Y_test, y_pred)
precision_score_xg = precision_score(Y_test, y_pred)
print("Sensitivity/Recall for XGBoost Model : {recall_score}".format(recall_score = recall_score_xg))
print("Precision of the XGBoost Model ",precision_score_xg)
print("F1 Score for XGBoost Model : {f1_score}".format(f1_score = f1_score_xg))


