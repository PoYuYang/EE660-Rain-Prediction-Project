import utils
from modules import run_model
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import pickle 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score ,plot_confusion_matrix
import modules


path = os.getcwd()
data = pd.read_csv(path+'/weatherAUS.csv',encoding="utf-8")

#replace label to binary value
data["RainToday"].replace({"No":0,"Yes":1},inplace = True)
data["RainTomorrow"].replace({"No":0,"Yes":1},inplace = True)


# filling nan to most appeared val
data['Date'] = data['Date'].fillna(data['Date'].mode()[0])
data['Location'] = data['Location'].fillna(data['Location'].mode()[0])
data['WindGustDir'] = data['WindGustDir'].fillna(data['WindGustDir'].mode()[0])
data['WindDir9am'] = data['WindDir9am'].fillna(data['WindDir9am'].mode()[0])
data['WindDir3pm'] = data['WindDir3pm'].fillna(data['WindDir3pm'].mode()[0])

# LabelEncode for non-numerical value
lcoder = {}
for col in data.select_dtypes(include = ["object"]).columns:
    lcoder[col] = LabelEncoder()
    data[col] = lcoder[col].fit_transform(data[col])

# balance data 
No = data[data.RainTomorrow == 0]
Yes = data[data.RainTomorrow == 1]
Yes_balance = resample(Yes,replace = True, n_samples = len(No),  random_state = 0)
N_data = pd.concat([No,Yes_balance])

# drop out extra features
drop_columns_list = ['WindGustSpeed','Pressure9am','MinTemp','MaxTemp','Date','RISK_MM']
N_data = N_data.drop(drop_columns_list, axis=1)

N_data = N_data.values
X = N_data[:,:-1]
Y = N_data[:,-1]
X.shape

# Split data to trian, val, test
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2, random_state = 123)
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size = 0.2, random_state = 123)

import warnings
warnings.filterwarnings("ignore")

# fill up missing data
imputer = IterativeImputer()
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
X_val = imputer.transform(X_val)


# Decision Tree
# dtree = tree.DecisionTreeClassifier(max_depth= 45)
print("Decision Tree")
with open('dtree.pickle', 'rb') as f:
    clf = pickle.load(f)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(acc)

# Random Forest
# rf = RandomForestClassifier(max_depth = 45,n_estimators = 70)
print("Random Forest")
with open('rf.pickle', 'rb') as f:
    clf = pickle.load(f)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(acc)

# Adaboost
# ada = AdaBoostClassifier(base_estimator = tree.DecisionTreeClassifier(max_depth=23), n_estimators=50, random_state=0)
print("Adaboost")
with open('ada.pickle', 'rb') as f:
    clf = pickle.load(f)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(acc)


# Logistic Regression
# lr = LogisticRegression(penalty = 'l1', solver = 'liblinear', random_state=123)
print("Logistic Regression")
with open('lr.pickle', 'rb') as f:
    clf = pickle.load(f)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(acc)


"""### Semi Supervised Learning """

X_train = X_train[:50000,:]
y_train = y_train[:50000]

# relabel some data to -1 for SSL
split = len(X_train)//2
X_1, X_2  = X_train[:split,:], X_train[split:,:]
y_1, y_2  = y_train[:split], y_train[split:]
y_2[:] = -1
y_1_2 = np.concatenate((y_1, y_2))
X_1_2 = np.concatenate((X_1, X_2))


#  Label Propagation
# lp = LabelPropagation(kernel='knn', n_neighbors= 20 , gamma=0, max_iter=5000, tol=0.0001)
print("Label Propagation")
with open('lp.pickle', 'rb') as f:
    clf = pickle.load(f)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(acc)

# Label Spreading 
# ls = LabelSpreading(kernel='knn', n_neighbors= 20, gamma=0, alpha= 0.03 , max_iter=5000, tol=0.0001)
print("Label Spreading")
with open('ls.pickle', 'rb') as f:
    clf = pickle.load(f)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(acc)

