# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
import warnings
warnings.filterwarnings('ignore')
# %%
file_path = 'C:\\Users\\Admin\Downloads\\telecom_customer_churn.csv'
df = pd.read_csv(file_path)
df
# %%
df.head()
# %%
df.shape
# %%
data_types = df.dtypes
data_types
# %%
df.info()
# %%
columns_to_drop = ['Internet Type', 'Avg Monthly GB Download', 'Online Security', 'Online Backup',
                   'Device Protection Plan', 'Premium Tech Support', 'Streaming TV', 'Streaming Movies',
                   'Streaming Music', 'Unlimited Data','Total Charges','Total Long Distance Charges','Zip Code',
                   'Latitude','Longitude']

df = df.drop(columns=columns_to_drop)
# %%
df
# %%
df.shape
# %%
df.isnull().sum()
# %%
cat = [i for i in dict(data_types) if dict(data_types)[i] == 'O']
num = [i for i in dict(data_types) if dict(data_types)[i] != 'O']
print(cat)
print(num)
# %%
"""
Imputing the Missing values 
"""
# For Columan 'OFFER'
offer_mode = df['Offer'].mode().iloc[0]
df['Offer'] = df['Offer'].fillna(value=offer_mode)
# %%
avg_month_charges_mean = df['Avg Monthly Long Distance Charges'].mean()
df['Avg Monthly Long Distance Charges'] = df['Avg Monthly Long Distance Charges'].fillna(value=avg_month_charges_mean)
# %%
df['Multiple Lines'].fillna(method='pad',inplace=True)
# %%
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')
df['Churn Category'] = df['Churn Category'].fillna(df['Churn Category'].mode()[0])
df['Churn Reason'] = df['Churn Reason'].fillna(df['Churn Reason'].mode()[0])
# %%
print(df.isnull().sum())
# %%
dataa_types = df.dtypes
dataa_types
# %%
cataegoreical = [i for i in dict(dataa_types) if dict(dataa_types)[i] == 'O']
numerical = [i for i in dict(dataa_types) if dict(dataa_types)[i] != 'O']
# %%
"""
Categorial data Analysis 
"""
cataegoreical
# %%
df.nunique()
# %%
"""
    Bar plots
"""
categorical_columns = cataegoreical
fig, axes = plt.subplots(nrows=len(categorical_columns)//2, ncols=2, figsize=(12, 18))
fig.subplots_adjust(hspace=0.5)
for i, column in enumerate(categorical_columns):
    row, col = i // 2, i % 2
    sns.countplot(x=column, data=df, ax=axes[row, col], palette='viridis')
    axes[row, col].set_title(f'Countplot of {column}')

plt.show()
# %%
"""
Pie Charts
"""
categorical_columns = cataegoreical
fig, axes = plt.subplots(nrows=len(categorical_columns)//2, ncols=2, figsize=(12, 18))
fig.subplots_adjust(hspace=0.5)
for i, column in enumerate(categorical_columns):
    row, col = i // 2, i % 2
    value_counts = df[column].value_counts()
    axes[row, col].pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
    axes[row, col].set_title(f'Pie Chart of {column}')
# %%
# Numerical Analysis
numerical
# %%
df.nunique()
# %%
df.describe()
# %%
# %%

numerical_columns = numerical
num_plots = len(numerical_columns)
num_rows = int(np.ceil(num_plots / 2))
fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(20,25))
fig.subplots_adjust(hspace=0.5)
for i, column in enumerate(numerical_columns):
    row, col = i // 2, i % 2
    axes[row, col].hist(df[column], bins=40, color='skyblue', edgecolor='black')
    axes[row, col].set_title(f'Histogram of {column}')
if num_plots % 2 != 0:
    fig.delaxes(axes[-1, -1])
plt.show()

# %%
"""
Checking for Outlaiers 
"""
# for the columan age
# %%
plt.title('Total Revenue')
plt.boxplot(df['Total Revenue'],vert=False)
plt.show()
# %%
revenue_df = df['Total Revenue']
revenue_df
# %%
def outlaiers_data():
    
    p1 = round(np.percentile(df['Total Revenue'],25),2)
    p2 = round(np.percentile(df['Total Revenue'],50),2)
    p3 = round(np.percentile(df['Total Revenue'],75),2)
    iqr = p3-p1
    ub = p3+1.5*iqr
    lb = p1-1.5*iqr
    cond1 = revenue_df>ub
    cond2 = revenue_df<lb
    outlaiers = revenue_df[cond1|cond2].values
    return outlaiers
outlaiers= outlaiers_data()
outlaiers_data() 
# %%
len(outlaiers_data())
# %%
def Non_outlaiers_data():
    q1 = round(np.percentile(df['Total Revenue'],25),2)
    q2 = round(np.percentile(df['Total Revenue'],50),2)
    q3 = round(np.percentile(df['Total Revenue'],75),2)
    iqr = q3-q1
    ub = q3+1.5*iqr
    lb = q1-1.5*iqr
    cond1 = revenue_df<ub
    cond2 = revenue_df>lb
    non_outlaiers = revenue_df[cond1&cond2]
    return non_outlaiers
non_outlaiers = Non_outlaiers_data()
Non_outlaiers_data()
# %%
len(Non_outlaiers_data())
# %%
len(outlaiers_data())+len(Non_outlaiers_data())
# %%
plt.subplot(1,2,1)
plt.title('Before outlaiers')
plt.boxplot(df['Total Revenue'])

plt.subplot(1,2,2)
plt.title('after outlaiers')
plt.boxplot(Non_outlaiers_data())
plt.show()
# %%
p1 = round(np.percentile(df['Total Revenue'],25),2)
p2 = round(np.percentile(df['Total Revenue'],50),2)
p3 = round(np.percentile(df['Total Revenue'],75),2)
iqr = p3-p1
ub = p3+1.5*iqr
lb = p1-1.5*iqr
cond1 = revenue_df>ub
cond2 = revenue_df<lb
# %%
df=df[df['Total Revenue']<ub]
# %%
df.shape
# %%
corr_df = df.corr(numeric_only=True)
corr_df
# %%
sns.heatmap(corr_df,annot=True)
# %%
plt.title('ScatterPlot')
plt.scatter(x=df['Total Revenue'],y=df['Tenure in Months'])
plt.show()
# %%
"""
Categorical data to numerical dat
"""
# %%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in cataegoreical:
    df[i]= le.fit_transform(df[i])
# %%
df
# %%
from sklearn.preprocessing import StandardScaler, LabelEncoder
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
df1 = df.copy()
label_encoder = LabelEncoder()
for column in df1.select_dtypes(include=['object']).columns:
    df1[column] = label_encoder.fit_transform(df1[column])
ss = StandardScaler()
df1[numerical_columns] = ss.fit_transform(df1[numerical_columns])
# %%
df1
# %%
df1.columns
# %%
X = df1.drop('Customer Status',axis=1)
# %%
y = df['Customer Status']
# %%
X.shape
# %%
y.shape
# %%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,
                                                 y,
                                                 random_state=42,
                                                 test_size=0.3)
# %%
X_train.shape,X_test.shape
# %%
y_train.shape,y_test.shape
# %%
X_train.head()
# %%
X_test.head()
# %%
y_train[:5]
# %%
