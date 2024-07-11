# -*- coding: utf-8 -*-
"""Heart Disease Prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1IbodtK2LilnxOs7FeJrI0coelAahBuUd
"""

from google.colab import drive
drive.mount('/content/drive')

"""Import the necessary libraries."""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

df = pd.read_csv('/content/drive/MyDrive/DATASETS/heart.csv')

df.head()



"""1. Preliminary analysis:
Perform preliminary data inspection to examine the structure of the data, missing values, duplicates.
"""

df.columns

df.info()

df.shape

df.describe()

df.duplicated().sum()

df.drop_duplicates(inplace=True)

df.isnull().sum()

"""EXPLORATORY DATA ANALYSIS"""



df.sex.value_counts()

"""Bivariate analysis of the target column wrt to the age column"""

sns.boxenplot(data=df, x='target', y='age')

#plt.figure(figsize=(15,15))
plt.rcParams['figure.figsize']=(20,9)
sns.heatmap(df.corr(),annot= True, fmt= '.2f')
plt.title("Correlation between variables")



sns.set_style("whitegrid")
sns.countplot(x='target',data=df, palette='RdBu_r')
plt.rcParams['figure.figsize']=(15,6)
plt.title('Count of target variable')



sns.pairplot(df)



"""MODEL TRAINING"""

X =df.iloc[:,:-1]
y=df.iloc[:,-1]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""USE LAZY PREDICT TO SELECT THE BEST 3 MODELS"""

!pip install lazypredict
from lazypredict.Supervised import LazyClassifier

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train_scaled, X_test_scaled, y_train, y_test)

models

"""LOGISTIC REGRESSION MODEL"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
y_predict= lr.predict(X_test_scaled)
lr_acc= accuracy_score(y_test,y_predict)
print("Accuracy of Logistic Regression classifier is ",lr_acc)
print('\n')

"""RANDOM FOREST CLASSIFIER"""

from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier()
rf = RandomForestClassifier(n_estimators=50, random_state=12,max_depth=5)
rf.fit(X_train_scaled,y_train)
rf_predicted = rf.predict(X_test_scaled)
rf_conf_matrix = confusion_matrix(y_test, rf_predicted)
rf_acc_score = accuracy_score(y_test, rf_predicted)
print("confusion matrix")
print(rf_conf_matrix)
print("\n")
print("Accuracy of Random Forest:",rf_acc_score*100,'\n')
print(classification_report(y_test,rf_predicted))

"""HYPERPARAMETER TUNING"""

grid_params = [
    {"n_estimators": [50,100,150,200,250,300],
    "criterion": ["gini", "entropy"],
    "max_features": ["auto", "sqrt", "log2"],
    },
]

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(estimator = rf, param_grid= grid_params, scoring = 'accuracy', n_jobs= -1, cv = 3, verbose = 10)

best_model = grid_search.fit(X_train_scaled,y_train.ravel())

best_model.best_params_

best_model.best_estimator_

"""RETRAIN WITH THE BEST PARAMETER FOR RANDOM FOREST CLASSIFIER"""

from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier()
rf = RandomForestClassifier(n_estimators=200, random_state=12,max_depth=5)
rf.fit(X_train_scaled,y_train)
rf_predicted = rf.predict(X_test_scaled)
rf_conf_matrix = confusion_matrix(y_test, rf_predicted)
rf_acc_score = accuracy_score(y_test, rf_predicted)
print("confusion matrix")
print(rf_conf_matrix)
print("\n")
print("Accuracy of Random Forest:",rf_acc_score*100,'\n')
print(classification_report(y_test,rf_predicted))

"""HYPERPAREMETER TUNING FOR LOGISTIC REGRESSION"""

param_grid = [
    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
     'C' : [0.001, 0.01, 0.1, 1, 10, 100],
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000,2500, 5000],
     }
]

lr_cv = GridSearchCV(estimator = lr, param_grid= param_grid, scoring = 'accuracy', n_jobs= -1, cv = 3, verbose = 10)

best_model = lr_cv.fit(X_train_scaled, y_train)

best_model.best_params_

best_model.best_estimator_

"""RETRAIN WITH THE BEST PARAMETER FOR LOGISTIC REGRESSION MODEL"""



lr = LogisticRegression(C=1, max_iter=100, penalty='l2', solver="liblinear")
lr.fit(X_train_scaled, y_train)
y_predict= lr.predict(X_test_scaled)
lr_acc= accuracy_score(y_test,y_predict)
print("Accuracy of Logistic Regression classifier is ",lr_acc)
print('\n')



"""USING KNEIGHBORS_CLASSIFIER MODEL"""

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train_scaled, y_train)
knn_predicted = knn.predict(X_test_scaled)
knn_conf_matrix = confusion_matrix(y_test, knn_predicted)
knn_acc_score = accuracy_score(y_test, knn_predicted)
print("confussion matrix")
print(knn_conf_matrix)
print("\n")
print("Accuracy of K-NeighborsClassifier:",knn_acc_score*100,'\n')
print(classification_report (y_test,knn_predicted))

error_rate=[]
for i in range(1,20):
    model1=KNeighborsClassifier(n_neighbors=i)
    model1.fit(X_train_scaled,y_train)
    k_y_test_pred=model1.predict(X_test_scaled)
    error_rate.append(np.mean(k_y_test_pred != y_test))



plt.rcParams['figure.figsize']=(15,6)
plt.plot(range(1,20),error_rate,color='purple',linestyle='dashed',marker='o',markerfacecolor='red', markersize=10)
plt.title("Error rate vs K value")
plt.xlabel("K values")
plt.ylabel("Error rate")
plt.show()

print("Minimum Value of error rate:-",min(error_rate),"obtain at K=",error_rate.index(min(error_rate)))

acc=[]
for i in range(1,20):
    model1=KNeighborsClassifier(n_neighbors=i)
    model1.fit(X_train_scaled,y_train)
    k_y_test_pred=model1.predict(X_test_scaled)
    acc.append(accuracy_score(k_y_test_pred,y_test))


plt.figure(figsize=(15,6))
plt.plot(range(1,20),acc,color='purple',linestyle='dashed',marker='o',markerfacecolor='red', markersize=10)
plt.title("Accuracy vs K value")
plt.xlabel("K values")
plt.ylabel("Accuracy rate")
plt.show()

print("Maximum accuracy:-",max(acc),"obtain at K=",acc.index(max(acc)))
print("\n Confusion Matrix:\n",confusion_matrix(k_y_test_pred,y_test))
kn_acc=max(acc)



"""RETRAIN WITH NEW K-VALUE

"""

knn = KNeighborsClassifier()
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
knn_predicted = knn.predict(X_test_scaled)
knn_conf_matrix = confusion_matrix(y_test, knn_predicted)
knn_acc_score = accuracy_score(y_test, knn_predicted)
print("confussion matrix")
print(knn_conf_matrix)
print("\n")
print("Accuracy of K-NeighborsClassifier:",knn_acc_score*100,'\n')
print(classification_report (y_test,knn_predicted))

"""MODEL EVALUATION"""

model_evaluation=pd.DataFrame({"model_name":["Logistic Regression","RandomForestClassifier","KNeighborsClassifier",
                                             ],
                             "Accuracy":[lr_acc*100,rf_acc_score*100,knn_acc_score*100]})
model_evaluation

"""**CONCLUSION**"""

colors = ['red','green','blue','gold','silver','yellow','orange',]
plt.figure(figsize=(16,5))
plt.title("barplot Represent Accuracy of different models")
plt.xlabel("Accuracy %")
plt.ylabel("Algorithms")
plt.bar(model_evaluation['model_name'],model_evaluation['Accuracy'],color = colors)
plt.show()

"""From the above evaluation KneighborsClassifier and RandomForestClassifier can be used as the machine learning algorithm to predict the heart disease of a patient"""

import pickle
with open("heart_disease.pkl","wb") as f:
  pickle.dump(rf,f)

import sklearn
print(sklearn.__version__)
