# %%
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns 
sns.set_theme(color_codes=True)
pd.set_option('display.max_columns',None)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,roc_auc_score,roc_curve,confusion_matrix,ConfusionMatrixDisplay
# %%
file_path = 'C:\\Users\\Admin\\Data Science\\ML\\churan data\\Preprocessed_data.csv'
data = pd.read_csv(file_path)
data

# %%
#step-1
#- Divide the data into input data and output data
# %%
X = data.drop('Customer Status',axis=1)
# %%
y = data['Customer Status']
# %%
X.shape,y.shape
# %%
#step -2
#- Train-Test-Split
# %%
# test size = 0.2 , train data 80% and test data 20%
# Random state = 42
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,
                                                 y,
                                                 random_state=42,
                                                 test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# %%
#STEP - 3
#- Removing the outlaiers using z-score
# %%
from scipy import stats
selcted_columns = ['Number of Dependents','Avg Monthly GB Download','Total Refunds'
                   ,'Total Extra Data Charges','Total Long Distance Charges','Total Revenue']
z_scores = np.abs(stats.zscore(X_train[selcted_columns]))
threshold = 3
outlier_indices  =np.where(z_scores>threshold)[0]
X_train = X_train.drop(X_train.index[outlier_indices])
y_train = y_train.drop(y_train.index[outlier_indices])
X_train.head()
# %%
#Without Hyperparameter tuning
# we are not any parameters
#we are using defult parameters
# %%
#DECISION TREE
# %%
#model develop
#predications
#evaluation
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
# %%
ydt_pred = dtree.predict(X_test)
print(ydt_pred[:10])
print(y_test.values[:10])
# %%
acc_dt = round(accuracy_score(y_test,ydt_pred),2)
acc_dt
# %%
confusion_matrix(y_test,ydt_pred)
# %%
tn,fp,fn,tp = confusion_matrix(y_test,ydt_pred).ravel()
# %%
print(tn)
print(fp)
print(fn)
print(tp)
# %%
print(accuracy_score(y_test,ydt_pred))
print(precision_score(y_test,ydt_pred))
print(recall_score(y_test,ydt_pred))
print(f1_score(y_test,ydt_pred))
# %%
cmt=confusion_matrix(y_test,ydt_pred)
ConfusionMatrixDisplay(cmt,display_labels=[False,True]).plot()
plt.grid(False)
# %%
ydt_pred
# %%
dtree.predict_proba(X_test)
# col=1   col=2
# col=1 related
# [0., 1.]
# when you pass one observation X_test === No Yes
# P(No)=0
# P(Yes)=1
# %%
# we need to extract only class-1 prob
ydt_pred_prob = dtree.predict_proba(X_test)[:,1]
roc_curve(y_test,ydt_pred_prob)
# %%
ydt_pred = dtree.predict(X_test)
accuracy_score(y_test,ydt_pred)
# %%
ydt_pred_prob=dtree.predict_proba(X_test)[:,1]
fpr,tpr,threshold = roc_curve(y_test,ydt_pred_prob)
plt.plot(fpr,tpr)
# %%
# Writing the above all code together
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

# ============================Ste-5 : Predictions =============
ydt_pred = dtree.predict(X_test)
# ============================= Step-6 : Metrics ================
acc_dt = round(accuracy_score(y_test,ydt_pred),2)
prec_dt = round(precision_score(y_test,ydt_pred),2)
recall_dt =round(recall_score(y_test,ydt_pred),2)
f1_dt = round(f1_score(y_test,ydt_pred),2)

print('accuracy is',acc_dt)
print('Precsion is',prec_dt)
print('recall is',recall_dt)
print('f1 score is ',f1_dt)
print(classification_report(y_test,ydt_pred))
# ===================================== Step-7 Confision Matrix ==========
cmt = confusion_matrix(y_test,ydt_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cmt,
                              display_labels=[False,True])
disp.plot()
plt.grid()
plt.show()


tn,fp,fn,tp = confusion_matrix(y_test,ydt_pred).ravel()
print('True Negatve is',tn)
print('False Positive is ',fp)
print('False Negative',fn)
print('True Positive',tp)


# ======================================== Step-8 : ROC-AUC curve ===========
ydt_pred_prob = dtree.predict_proba(X_test)[:,1]
fpr,tpr,threshold= roc_curve(y_test,ydt_pred_prob)
plt.plot(fpr,tpr)
# %%
# Logistic Regressoin
from sklearn.linear_model import LogisticRegression
logtree = LogisticRegression()
logtree.fit(X_train,y_train)

# ================================== Step-5 : Predictions ==========
ylog_pred =logtree.predict(X_test)

# =========================== Step-6 : Metrics ==========
acc_log = round(accuracy_score(y_test,ylog_pred),2)
prec_log = round(precision_score(y_test,ylog_pred),2)
recall_log = round(recall_score(y_test,ylog_pred),2)
f1_log = round(f1_score(y_test,ylog_pred),2)
print('Accuracy is ',acc_log)
print('Precision is ',prec_log)
print('Recall is ',recall_log)
print('f1 score is ',f1_log)

# ============================ Step-7 : Confusion Matrix ========
cmt = confusion_matrix(y_test,ylog_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cmt,
                              display_labels=[False,True])
disp.plot()
plt.grid(False)
plt.show()


tn,fp,fn,tp = confusion_matrix(y_test,ylog_pred).ravel()
print('True Negative:',tn)
print('False Positive:',fp)
print('False Negative :',fn)
print('True Positive :',tp)
# ============================ Steo-8 : ROC-AUC curve ======
ylog_pred_prob = logtree.predict_proba(X_test)[:,1]
fpr,tpr,threshold = roc_curve(y_test,ylog_pred_prob)
plt.plot(fpr,tpr)
plt.show()
# %%
# Naive Bayes
# ======================== Step-4 : Train the model ==
from sklearn.naive_bayes import GaussianNB
nbtree = GaussianNB()
nbtree.fit(X_train,y_train)

# ===================== Step-5 : Predictions ===========
ynb_pred = nbtree.predict(X_test)
# ===================== Step-6 : Metrices ===================
acc_nb = round(accuracy_score(y_test,ynb_pred),2)
prec_nb = round(precision_score(y_test,ynb_pred),2)
rec_nb = round(recall_score(y_test,ynb_pred),2)
f1_nb = round(f1_score(y_test,ynb_pred),2)
print('Accuracy is ',acc_nb)
print('Precison is ',prec_nb)
print('Recall is',rec_nb)
print('f1_score',f1_nb)
# ========================= Step-7 : Confusion matirx ======== 
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
cmt = confusion_matrix(y_test,ynb_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cmt,
                              display_labels=[False,True])
disp.plot()
plt.grid()
plt.show()



tn,fp,fn,tp = confusion_matrix(y_test,ynb_pred).ravel()
print('True Negative : ',tn)
print('Flase Positive : ',fp)
print('False Negative :',fn)
print("True Postive : ",tp)


# ====================================Step-9 : ROC-AUC : curve ======
ynb_pred_prob = nbtree.predict_proba(X_test)[:,1]
fpr,tpr,threshold = roc_curve(y_test,ynb_pred_prob)
plt.plot(fpr,tpr)
plt.show()
# %%
# KNN
# =================== Step-4: train the model ====
from sklearn.neighbors import KNeighborsClassifier
knntree = KNeighborsClassifier()
knntree.fit(X_train,y_train)

# ================== Step-5 : Predictiond ====

yknn_pred = knntree.predict(X_test)

# ================== Step-6 : Metrics =============
acc_knn = round(accuracy_score(y_test,yknn_pred),2)
prec_knn = round(precision_score(y_test,yknn_pred),2)
rec_knn = round(recall_score(y_test,yknn_pred),2)
f1_knn = round(f1_score(y_test,yknn_pred),2)
print('Accuracy is ',acc_knn)
print('Precision is ',prec_knn)
print('Recall is ',rec_knn)
print('F1 score ',f1_knn)


# ===================== Step-7 :Confusion Metrix =====
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
cmt =confusion_matrix(y_test,yknn_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cmt,
                              display_labels=[False,True])
disp.plot()
plt.grid()
plt.show()



tn,fp,fn,tp = confusion_matrix(y_test,yknn_pred).ravel()

print('True Negative : ',tn)
print('Flase Positive : ',fp)
print('False Negative : ',fn)
print('True Positive : ',tp)

# ======================== Step -8 : ROC-AUC curve ===

yknn_pred_prob = knntree.predict_proba(X_test)[:,1]
fpr,tpr,threshold = roc_curve(y_test,yknn_pred_prob)
plt.plot(fpr,tpr)
plt.show()
# %%
# Random Forest
# ========================== Step-4 : Train the model  =====
from sklearn.ensemble import RandomForestClassifier
rftree = RandomForestClassifier()
rftree.fit(X_train,y_train)

# ========================= Step-5 : Predictions ====
yrf_pred = rftree.predict(X_test)

# ========================== Step-6 : Metrices =========
acc_rf = round(accuracy_score(y_test,yrf_pred),2)
prec_rf = round(precision_score(y_test,yrf_pred),2)
recall_rf = round(recall_score(y_test,yrf_pred),2)
f1_rf = round(f1_score(y_test,yrf_pred),2)

print('Accuracy is ',acc_rf)
print('Precsion is ',prec_rf)
print('Recall is',recall_rf)
print('F1 Score ',f1_rf)


#====================== Step-7 : Confusion Matrix ======
cmt = confusion_matrix(y_test,yrf_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cmt,
                              display_labels=[False,True])
disp.plot()
plt.grid()
plt.show()



tn,fp,fn,tp = confusion_matrix(y_test,yrf_pred).ravel()


print('True Negative is ',tn)
print('False Positive is ',fp)
print('False Nehative is ',fn)
print('True Ppsitive is',tp)

# ========================== Step-8 : ROC-AUC curve ==========
yrf_pred_prob = rftree.predict_proba(X_test)[:,1]
fpr,tpr,threshold = roc_curve(y_test,yrf_pred_prob)
plt.plot(fpr,tpr)
plt.show()
# %%
dict = {'Accuraacy':[acc_dt,acc_log,acc_nb,acc_knn,acc_rf],
        'Precision':[prec_dt,prec_log,prec_nb,prec_knn,prec_rf],
        'Recall':[recall_dt,recall_log,rec_nb,rec_knn,recall_rf],
        'F1 Score': [f1_dt,f1_log,f1_nb,f1_knn,f1_rf]}
pd.DataFrame(dict,index=['DT','LR','NB','KNN','Random Forest'])
# %%
