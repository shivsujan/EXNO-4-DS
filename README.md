# EXNO:4-DS

# AIM:

To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:

STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:

1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1

2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.

3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.

4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:

Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```
import pandas as pd
from scipy import stats
import numpy as np
```
```
df=pd.read_csv("/content/bmi.csv")
df.head()
```
<img width="351" height="233" alt="image" src="https://github.com/user-attachments/assets/71ef3b6f-97c1-4a62-8ab4-c46ec4c7a3c6" />

```
df_null_sum=df.isnull().sum()
df_null_sum
```
<img width="198" height="189" alt="image" src="https://github.com/user-attachments/assets/28397ddb-0459-4c40-9dfb-b23e1e1d494b" />

```
df.dropna()   
```
<img width="374" height="387" alt="image" src="https://github.com/user-attachments/assets/321170c3-fdf7-4f93-8af9-c81710e54b99" />

```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
```
<img width="164" height="133" alt="image" src="https://github.com/user-attachments/assets/337bcf85-326e-4a35-a631-9bcdcb813eee" />

```
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()
```
<img width="364" height="188" alt="image" src="https://github.com/user-attachments/assets/be50a043-66d9-48ac-a25c-fb58543af317" />

```
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
<img width="381" height="642" alt="image" src="https://github.com/user-attachments/assets/86878121-ceaa-4aa5-972e-c2b5047df141" />

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
<img width="367" height="328" alt="image" src="https://github.com/user-attachments/assets/8a71cb58-2d62-4c05-b04e-6e4e68ada738" />

```
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
<img width="396" height="385" alt="image" src="https://github.com/user-attachments/assets/2ff2f059-2628-4a60-892d-74e20af9398c" />

```
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3.head()
```
<img width="385" height="186" alt="image" src="https://github.com/user-attachments/assets/eb6fc863-931f-4a34-b450-5071bb11546b" />

```
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```
<img width="400" height="329" alt="image" src="https://github.com/user-attachments/assets/5e8e38ab-3257-4830-a44d-d0a17aba10b2" />

```
df_null_sum=df.isnull().sum()
df_null_sum
```
<img width="185" height="442" alt="image" src="https://github.com/user-attachments/assets/06413c53-6244-42f1-98c4-0be55f786ce7" />

```
# Chi_Square
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
#In feature selection, converting columns to categorical helps certain algorithms
# (like decision trees or chi-square tests) correctly understand and
 # process non-numeric features. It ensures the model treats these columns as categories,
  # not as continuous numerical values.
df[categorical_columns]
```
<img width="830" height="377" alt="image" src="https://github.com/user-attachments/assets/b664b10a-a658-4c73-aa26-e58cd5d88a1c" />

```
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
##This code replaces each categorical column in the DataFrame with numbers that represent the categories.
df[categorical_columns]
```
<img width="737" height="376" alt="image" src="https://github.com/user-attachments/assets/f9deb32b-0d78-400f-977b-0606151a90d6" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```
```
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
<img width="344" height="73" alt="image" src="https://github.com/user-attachments/assets/f35b9f7e-bc80-466c-bbb8-197fe852ac3d" />

```
y_pred = rf.predict(X_test)
```
```
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```
<img width="376" height="323" alt="image" src="https://github.com/user-attachments/assets/47d7c027-5ab1-47e9-bea7-8e46ca51c313" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
<img width="824" height="374" alt="image" src="https://github.com/user-attachments/assets/f4c500d4-ca7f-4f23-bf7f-6cf34366cd15" />

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
<img width="733" height="372" alt="image" src="https://github.com/user-attachments/assets/18e5d80b-82e6-4eb6-b742-9d95ac76fcaf" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```
<img width="595" height="80" alt="image" src="https://github.com/user-attachments/assets/c50940de-3119-4961-b2b6-f4a2e370e562" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split # Importing the missing function
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
<img width="364" height="69" alt="image" src="https://github.com/user-attachments/assets/c1947d2d-3db7-4ab8-9627-e3921048d3d7" />

```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
<img width="473" height="30" alt="image" src="https://github.com/user-attachments/assets/ed82ab09-370d-44d7-b299-3bf70e3ab688" />

```
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```
```
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]

df[categorical_columns] = df[categorical_columns].astype('category')
```
```
  df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
  # @title
  df[categorical_columns]
```
<img width="813" height="380" alt="image" src="https://github.com/user-attachments/assets/af94108c-53be-426f-9396-45edfc8b8fe6" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```
```
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
```
```
selected_features_anova = X.columns[selector_anova.get_support()]
```
```
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```
<img width="670" height="60" alt="image" src="https://github.com/user-attachments/assets/9287f175-7ec1-4fe0-87d8-f887262608ce" />


```
# Wrapper Method
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("/content/income(1) (1).csv")
# List of categorical columns
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]

# Convert the categorical columns to category dtype
df[categorical_columns] = df[categorical_columns].astype('category')
```
```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
```
```
df[categorical_columns]
```
<img width="775" height="377" alt="image" src="https://github.com/user-attachments/assets/2546c7bd-789f-4e30-bcde-d4a037735cc7" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```
```
logreg = LogisticRegression()
```
```
n_features_to_select =6
```
```
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```
# RESULT:

Therefore the codes are executed successfully
