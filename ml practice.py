# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
pd.set_option('display.max_columns',None)
sns.set_theme(color_codes=True)
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV,cross_val_score
# %%
file_path = 'C:\\Users\\Admin\\Data Science\\Real Time projects\\Telecom_Customer_Churn\\Preprocessed_data.csv'
df = pd.read_csv(file_path)
df
# %%
X = df.drop('Customer Status',axis=1)
# %%
y = df['Customer Status']
# %%
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.tree import DecisionTreeClassifier
gridtree = DecisionTreeClassifier()
gridtree
# %%
gridtree.get_params()
# %%
param_grid = {'criterion' :['gini','entropy'],
              'max_depth' : [3,4,5,6,7,8] ,
              'min_samples_leaf' : [2,3,4],
              'min_samples_split' : [1,2,3,4],
              'random_state' : [0,42]}
# %%
grid_search = GridSearchCV(gridtree,param_grid,
                           scoring='accuracy',
                           cv = 5,
                           verbose=True)
# %%
grid_search
# %%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,
                                                 y,
                                                 random_state=0,
                                                 test_size=0.2)
# %%
import time
start = time.time()
grid_search.fit(X_train,y_train)
end = time.time()
print('The time taken =',(end-start))
# %%
grid_search.best_estimator_
# %%
grid_search.best_params_
# # %%
grid_search.best_score_

# %%
# - Cross Validation score
accuracy_list = cross_val_score(grid_search.best_estimator_,
                                X_train,
                                y_train,
                                scoring='accuracy')
# %%
accuracy_list
# %%
accuracy_list.mean()
# %%
# - Identfy the Predctions and metrics using parameters
# - out of 1440 fittings one best fit model in outcome
# - our goal is by using above parameter we need to find the predictions
# - and we will evalutae model performance
# - for that import descision treee nd pass the best parameters
# - now fit the model with x_train and y_train
# - get the predictions by passsing the X_test
# - Comapre the predictions with y_test
# - Calculate
# - Accuracy 
# - Precsision 
# - Recall 
# - f1_score
# - Roc-Auc curve
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion='entropy',
                               max_depth=6,
                               min_samples_leaf=3,
                               min_samples_split=2,
                               random_state=0)
dtree.fit(X_train,y_train)

ydt_pred = dtree.predict(X_test)

# =====================metrics===============
acc_dt =round(accuracy_score(y_test,ydt_pred)*100,2)
prec_dt = round(precision_score(y_test,ydt_pred)*100,2)
recal_dt = round(recall_score(y_test,ydt_pred)*100,2)
f1_dt = round(f1_score(y_test,ydt_pred)*100,2)
print('Accuracy',acc_dt)
print('Precision',prec_dt)
print('recall',recal_dt)
print('f1 score',f1_dt)

# =================== confusuon matrix =========
cmt = confusion_matrix(y_test,ydt_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cmt,display_labels=[False,True])
disp.plot()
plt.grid()
plt.show()
# =======================
ydt_pred_proba = dtree.predict_proba(X_test)[:,1]
from sklearn.metrics import roc_curve
fpr,tpr,treshold = roc_curve(y_test,ydt_pred_proba)
plt.plot(fpr,tpr)
plt.show()
# %%
dtree.feature_importances_
# %%
imp_df = pd.DataFrame({
    'Feature_name':X_train.columns,
    'Imortance':dtree.feature_importances_
})
fi = imp_df.sort_values(by='importance',ascending=False)
fi
# %%
