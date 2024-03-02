#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[82]:


dataset = pd.read_csv("C:/Users/llift/OneDrive/Desktop/3.2/github/star_classification.csv")


# In[83]:


print(dataset.head(10))


# In[11]:





# In[87]:


import pandas as pd
import numpy as np
# Check class distribution
class_distribution = dataset['class'].value_counts()
print("Class Distribution Before Balancing:")
print(class_distribution)

# Determine the majority class
majority_class = class_distribution.idxmax()
minority_class = class_distribution.idxmin()
# Get the number of samples in the majority class
majority_class_count = class_distribution[majority_class]
minority_class_count = class_distribution[minority_class]
# Find indices of majority class samples
majority_indices = dataset[dataset['class'] == majority_class].index

# Randomly select majority class samples to drop
drop_indices = np.random.choice(majority_indices, size=majority_class_count - minority_class_count, replace=False)

# Drop selected majority class samples
balanced_df = dataset.drop(drop_indices)

# Check class distribution after balancing
balanced_class_distribution = balanced_df['class'].value_counts()
print("\nClass Distribution After Balancing:")
print(balanced_class_distribution)


# In[91]:


le = LabelEncoder()
le1 = le.fit_transform(balanced_df['class'])


# In[92]:


from sklearn.preprocessing import LabelEncoder
lebel = LabelEncoder()
lebelencode = lebel.fit_transform(balanced_df['class'])
balanced_df['class'] = lebelencode


# In[93]:


balanced_df['class']


# In[94]:


X = balanced_df[['obj_ID','alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'run_ID',
       'rerun_ID', 'cam_col', 'field_ID', 'spec_obj_ID', 'redshift',
       'plate', 'MJD', 'fiber_ID']]
Y = balanced_df[['class']]


# In[95]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3, 
                                                 random_state=42,
                                                 shuffle 
                                                 = True,stratify= Y)


# In[96]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# In[97]:


import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# In[98]:


model = LogisticRegression()
model.fit(X_train_std,Y_train)


# In[99]:


model.score(X_test_std,Y_test)


# In[100]:


coefficients = model.coef_


# In[101]:


print(coefficients)


# In[102]:


# Assuming coefficients is your array
coefficients = np.array([[ 2.56204307e-02, -8.20590589e-03, -4.96072520e-02, 6.45704147e+00,
                           5.98922276e+00, 5.61055467e-01, -1.01860004e+00, -1.88944441e+00,
                           2.56241128e-02, 0.00000000e+00, -3.79940768e-02, 2.28999222e-02,
                           5.35656296e-01, 1.98372620e+01, 5.35658866e-01, -1.06259750e+00,
                           1.58195113e-02],
                         [ 1.82605522e-02, 1.10349625e-01, 1.90806512e-01, -8.47177248e+00,
                           -3.32182385e+00, -2.58644852e+00, 2.27083915e+00, 6.50276964e+00,
                           1.82627867e-02, 0.00000000e+00, -2.29905841e-02, 6.75351104e-03,
                           4.89997596e-02, 2.38030912e+01, 4.89989614e-02, -2.34850214e-01,
                           4.73601078e-02],
                         [-4.38809829e-02, -1.02143719e-01, -1.41199260e-01, 2.01473102e+00,
                           -2.66739890e+00, 2.02539305e+00, -1.25223911e+00, -4.61332522e+00,
                           -4.38868995e-02, 0.00000000e+00, 6.09846609e-02, -2.96534333e-02,
                           -5.84656056e-01, -4.36403532e+01, -5.84657828e-01, 1.29744771e+00,
                           -6.31796191e-02]])

# Feature names (replace these with your actual feature names)
feature_names = ['obj_ID','alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'run_ID',
       'rerun_ID', 'cam_col', 'field_ID', 'spec_obj_ID', 'redshift',
       'plate', 'MJD', 'fiber_ID']

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
index = np.arange(len(feature_names))

for i, label in enumerate(range(len(coefficients))):
    ax.plot(index, coefficients[label], marker='o', label=f'Class {i + 1}')

ax.set_xlabel('Features')
ax.set_ylabel('Coefficients')
ax.set_title('Coefficient Values for Each Feature')
ax.set_xticks(index)
ax.set_xticklabels(feature_names, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.show()


# In[105]:


balanced_df_important_feature_only = balanced_df[['u', 'g', 'r', 'i', 'z', 'redshift','class']]


# In[106]:


df = pd.DataFrame(balanced_df_important_feature_only)

# Specify the path where you want to save the CSV file
csv_file_path = 'balanced_df_important_feature_only.csv'

# Export the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False)

