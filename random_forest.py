# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
pd.set_option('display.max_columns',None)
sns.set_theme(color_codes=True)
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_curve
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV,cross_val_score
# %%
file_path = 'C:\\Users\\Admin\\Data Science\\Real Time projects\\Telecom_Customer_Churn\\Preprocessed_data.csv'
df = pd.read_csv(file_path)
df
# %%
X = df.drop('Customer Status',axis=1)
y = df['Customer Status']
# %%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y,random_state=0,test_size=0.2)

# %%
from sklearn.ensemble import RandomForestClassifier
gridrf = RandomForestClassifier()
gridrf 
# %%
gridrf.get_params()
# %%
gridrf_param = {'n_estimators':[100,200],
                'criterion' : ['gini','entropy'],
                'max_depth' : [3,5,10],
                'max_features' : ['sqrt','log2'],
                'random_state' : [0,42]} 
# %%
grid_search = GridSearchCV(gridrf,gridrf_param,
                           cv=5,
                           verbose=True)
# %%
grid_search
# %%
import time
start = time.time()
grid_search.fit(X_train,y_train)
end = time.time()
print('the time taken = ',(end-start))
# %%
grid_search
# %%
