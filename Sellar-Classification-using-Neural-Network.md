# Sellar-Classification-using-Neural-Network
##
#
# Why This Project
#
#
#

# Dataset source
### [Kaggle](https://www.kaggle.com/datasets/deepu1109/star-dataset?fbclid=IwAR0k1NtV3FqDQLSlkyIyxM5QK4EjRpX8e66ohga19P20ED1hj0fEfRJL8EM)
![dataset-cover](https://github.com/640710505/Sellar-Classification-using-Neural-Network/assets/141728733/127c1383-0253-4732-a550-86f78801fb1a)
#### Star dataset to predict star types A 6 class star dataset for star classification with Deep Learned approaches 
# Dataset describe

#
#
#
#
#
# 
#
# Data Visualization
![Unknown](https://github.com/640710505/Sellar-Classification-using-Neural-Network/assets/114089025/e5c9f836-24ab-4d1e-b257-d0c5f782256a)
# 
#
#
# 
#
#
# Feature important 
### With Logistic Regression

```python
### All feature and Target(class)  
X = data[['obj_ID','alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'run_ID',
       'rerun_ID', 'cam_col', 'field_ID', 'spec_obj_ID', 'redshift',
       'plate', 'MJD', 'fiber_ID']]
Y = data[['class']]
```
```python
###  use train_test_split  by train 70 % and test 30%
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3, 
                                                 random_state=42,
                                                 shuffle 
                                                 = True,stratify= Y)
```
```python
### Use StandardScaler for scaling data to samesame  value 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test) 
```
```python
### test Model
model = LogisticRegression()
model.fit(X_train_std,Y_train)
```

![featureimportant](https://github.com/640710505/Sellar-Classification-using-Neural-Network/assets/141728733/10905af0-4587-4fcf-bc27-d759f84fc7d1)
# 
#
#
# Feature Extraction
#
#
# 



#
#
# 
#
# Neural Network Model
# 
#
#
# 
#
#
# 
#
# Score Analysis
# 


