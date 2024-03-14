# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(color_codes=True)             # To get diffent different colors
pd.set_option('display.max_columns', None) 
# %%
file_path = 'C:\\Users\\Admin\\Data Science\\Real Time projects\\Telecom_Customer_Churn\\Preprocessed_data.csv'
df = pd.read_csv(file_path)
df
# %%
X = df.drop('Customer Status',axis=1)
y = df['Customer Status']
# %%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,
                                                 y,
                                                 random_state=1234,
                                                 test_size=0.3)
# %%
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.tree import DecisionTreeClassifier
gridtree = DecisionTreeClassifier()
gridtree
# %%
gridtree.get_params()
# %%
param_grid = {
    'criterion' : ['gini','entrophy'],
    'max_depth' : [3,4,5,6,7,8],
    'min_samples_split' : [2,3,4],
    'min_samples_leaf' : [1,2,3,4],
    'random_state' : [0,42]
}

# %%
import time
start = time.time()
grid_search = GridSearchCV(gridtree,
                           param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=True)
end = time.time()
print((end-start))
# %%
grid_search
# %%
strat =  time.time()
grid_search.fit(X_train,y_train)
end = time.time()
print('the time take is',(end-start))
# %%
grid_search.best_estimator_
# %%
grid_search.best_score_
# %%
grid_search.best_params_
# %%
