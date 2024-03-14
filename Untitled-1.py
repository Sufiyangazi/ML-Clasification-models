# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_theme(color_codes=True)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,roc_auc_score,roc_curve
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
# %%
file_path = 'C:\\Users\\Admin\\Data Science\\ML\\churan data\\Preprocessed_data.csv'
df = pd.read_csv(file_path)
df
# %%
X = df.drop('Customer Status',axis=1)
y = df['Customer Status']
# %%
from sklearn.model_selection import train_test_split
# %%
X_train,X_test,y_train,y_test = train_test_split(X,
                                                 y,
                                                 random_state=42,
                                                 test_size=0.2)
# %%
from scipy import stats

# Define the columns for which you want to remove outliers
selected_columns = ['Number of Dependents', 'Avg Monthly GB Download', 'Total Refunds',
                    'Total Extra Data Charges', 'Total Long Distance Charges', 'Total Revenue']

# Calculate the Z-scores for the selected columns in the training data
z_scores = np.abs(stats.zscore(X_train[selected_columns]))

# Set a threshold value for outlier detection (e.g., 3)
threshold = 3

# Find the indices of outliers based on the threshold
outlier_indices = np.where(z_scores > threshold)[0]

# Remove the outliers from the training data
X_train = X_train.drop(X_train.index[outlier_indices])
y_train = y_train.drop(y_train.index[outlier_indices])
# %%
""""
without hyperparamter tuning
- Decision Tree
- KNN
- Naive Bayes
- Logistic regression
- Random Forest 
"""
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
# %%
dt_pred = dtree.predict(X_test)
dt_pred
# %%
print(dt_pred[:5])
print(y_test.values[:5])
# %%
print(dt_pred[:10])
print(y_test.values[:10])
# %%
accuracy_score(y_test,dt_pred)
# %%
confusion_matrix(y_test,dt_pred)
# %%
tn,fp,fn,tp = confusion_matrix(y_test,dt_pred).ravel()
# %%
print(tn,fp,fn,tp)
# %%
acc = (tn+tp)/(tn+tp+fp+fn)
print(acc)
# %%
prec = tp/(tp+fp)
print(prec)
# %%
rec = tp/(tp+fn)
print(rec)
# %%
f1 = (2*prec*rec)/(prec+rec)
print(f1)
# %%
precision_score(y_test,dt_pred)
# %%
recall_score(y_test,dt_pred)
# %%
f1_score(y_test,dt_pred)
# %%
cmt = confusion_matrix(y_test,dt_pred)
# %%
ConfusionMatrixDisplay(cmt).plot()
plt.grid(False)
# %%
"""
Roc-Auc Curve 
- X-Axis : FPR
- Y-Axis : TPR
- 
"""
dtree.predict_proba(X_test)
# %%
dt_pred
# %%
y_dt_predict_prob=dtree.predict_proba(X_test)[:,1]
y_dt_predict_prob
# %%
fpr,tpr,threshold=roc_curve(y_test,y_dt_predict_prob)
plt.plot(fpr,tpr)
# %%
