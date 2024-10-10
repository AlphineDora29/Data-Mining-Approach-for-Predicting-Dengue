#======================= IMPORT PACKAGES =============================

import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


#===================== DATA SELECTION ==============================

#=== READ A DATASET ====

data_frame=pd.read_csv("Dengue_dataset.csv")
print("--------------------------------- ---------------------")
print("================ 1. Data Selection ===================")
print("------------------------------------------------------")
print()
print(data_frame.head(20))

data_frame.hist()

#======================= PREPROCESSING =========================

#==== CHECKING MISSIMG VALUES =====
 
print("-----------------------------------------------------------")
print("=============== 2. Checking Missing Values ================")
print("-----------------------------------------------------------")
print()
print(data_frame.isnull().sum())


#===================== FEATURE SELECTION ==================================


x=data_frame.drop('target',axis=1)
y=data_frame['target']

print("====================================================")
print("----------------- 3. Chi-Square --------------------")
print("====================================================")
print()


chi2_features = SelectKBest(chi2, k = 10)
x_kbest= chi2_features.fit_transform(x, y)

print("Total no of original Features :",x.shape[1])
print()
print("Total no of reduced Features  :",x_kbest.shape[1])
print()


#===================== DATA SPLITTING ====================================

#==== TEST AND TRAIN ====


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

print("====================================================")
print("---------------- 4. Data Splitting -----------------")
print("====================================================")
print()
print("Total number of data's in input         :",data_frame.shape[0])
print()
print("Total number of data's in training part :",X_train.shape[0])
print()
print("Total number of data's in testing part  :",X_test.shape[0])
print()


#===================== CLASSIFICATION====================================


#=== RANDOM FOREST ===

from sklearn.ensemble import RandomForestClassifier

#initialize the model
rf = RandomForestClassifier(n_estimators = 10) 
 
#fitting the model
rf.fit(X_train, y_train) 

#predict the model
Y_pred_rf = rf.predict(X_test)

print("=================================================")
print("----------------- 1.Random Forest  --------------")
print("=================================================")
print()

accu_rf= metrics.accuracy_score(Y_pred_rf,y_test)*100

cm_rf=metrics.confusion_matrix(y_test,Y_pred_rf) 


print("1.Accuracy: ",accu_rf ,'%')
print() 
print("2.Confusion Matrix: ",cm_rf )
print()
print(metrics.classification_report(y_test, Y_pred_rf))
print()


#=== LOGISTIC REGRESSION ===

from sklearn import linear_model

from sklearn import metrics

lr=linear_model.LogisticRegression()


lr.fit(X_train, y_train)
 
pred_lr=lr.predict(X_test)

cm_lr=metrics.confusion_matrix(y_test, pred_lr)

acc_lr=metrics.accuracy_score(y_test, pred_lr)*100

print("====================================================")
print("-------------- 2.Logistic Regression ---------------")
print("===================================================")
print()


print("1.Accuracy: ",acc_lr ,'%')
print()
print("2.Confusion Matrix: ",cm_lr )
print()
print(metrics.classification_report(y_test, pred_lr))
print()


#=========================  PREDICTION ====================================


print("=================================================")
print("----------------- Prediction  -------------------")
print("=================================================")
print()
print()

for i in range(0,10):
    if pred_lr[i]==0:
        print([i],"=======================")
        print()
        print("       Normal          ")
    elif pred_lr[i]==1:
        print("=======================")
        print()
        print([i],"    Dengue---> Mild    ")
    else :
        print("=======================")
        print()
        print([i],"   Dengue---> Severe   ")



#========================== VISUALIZATION ==============================

import matplotlib.pyplot as plt
import numpy as np

objects = ('Random Forest', 'Logistic Regression')
y_pos = np.arange(len(objects))
performance = [accu_rf,acc_lr]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Performance ')
plt.title('Comparison Graph')
plt.show()
print()

from sklearn.metrics import confusion_matrix
import seaborn as sns
cf_matrix=confusion_matrix(y_test, pred_lr)
print(cf_matrix)
print()

sns.heatmap(cf_matrix,annot=True)
plt.title("Confusion Matrix")
plt.show()
print()

data_frame.iloc[14].plot(kind='bar')


data_frame.plot.bar()


