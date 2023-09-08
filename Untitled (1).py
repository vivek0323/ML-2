#!/usr/bin/env python
# coding: utf-8

# In[2]:


#A1


# In[3]:


import pandas as pd
import numpy as np

file_name = r"19CSE305_LabData_Set3.1.xlsx"
worksheet_name = 'thyroid0387_UCI'
data = pd.read_excel(file_name, sheet_name=worksheet_name)

data_types = data.dtypes

categorical_cols = data.select_dtypes(include=['object']).columns
nominal_cols = ['referral source'] + [col for col in data.columns if data[col].dtype == 'O' and data[col].str.contains('\?').any()]
ordinal_cols = list(set(categorical_cols) - set(nominal_cols))

numeric_cols = data.select_dtypes(include=['number'])
data_range = numeric_cols.describe().loc[['min', 'max']]

missing_values = data.isin(['?']).sum()

outliers = {}
for col in numeric_cols.columns:
    mean = numeric_cols[col].mean()
    std = numeric_cols[col].std()
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    outliers[col] = len(numeric_cols[(numeric_cols[col] < lower_bound) | (numeric_cols[col] > upper_bound)])

numeric_mean = numeric_cols.mean()
numeric_variance = numeric_cols.var()

print("Data Types:")
print(data_types)

print("\nEncoding Schemes:")
print("Nominal Columns:", nominal_cols)
print("Ordinal Columns:", ordinal_cols)

print("\nData Range:")
print(data_range)

print("\nMissing Values:")
print(missing_values)

print("\nOutliers:")
print(outliers)

print("\nMean and Variance for Numeric Variables:")
print("Mean:")
print(numeric_mean)
print("\nVariance:")
print(numeric_variance)


# In[4]:


#A2


# In[5]:


import pandas as pd
import numpy as np

file_name = r"19CSE305_LabData_Set3.1.xlsx"
worksheet_name = 'thyroid0387_UCI'
data = pd.read_excel(file_name, sheet_name=worksheet_name)

data.replace('?', np.nan, inplace=True)

for col in data.columns:
    if data[col].dtype == 'float64' or data[col].dtype == 'int64':
        if col in ['TSH', 'T3', 'TT4', 'T4U', 'FTI']:
            median_value = data[col].median()
            data[col].fillna(median_value, inplace=True)
        else:
            mean_value = data[col].mean()
            data[col].fillna(mean_value, inplace=True)
    elif data[col].dtype == 'object':
        mode_value = data[col].mode().iloc[0]
        data[col].fillna(mode_value, inplace=True)

missing_values_after_imputation = data.isnull().sum()

print("Missing Values After Imputation:")
print(missing_values_after_imputation)


# In[6]:


#A3


# In[8]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

file_name = r"19CSE305_LabData_Set3.1.xlsx"
worksheet_name = 'thyroid0387_UCI'
data = pd.read_excel(file_name, sheet_name=worksheet_name)

data.replace('?', np.nan, inplace=True)

numeric_attributes = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']

data[numeric_attributes] = data[numeric_attributes].apply(pd.to_numeric, errors='coerce')

minmax_scaler = MinMaxScaler()
data[numeric_attributes] = minmax_scaler.fit_transform(data[numeric_attributes])

print("Normalized Data:")
print(data.head())


# In[9]:


#A4


# In[12]:


import pandas as pd

file_name = r"19CSE305_LabData_Set3.1.xlsx"
worksheet_name = 'thyroid0387_UCI'
data = pd.read_excel(file_name, sheet_name=worksheet_name)

binary_attributes = ['on thyroxine', 'query on thyroxine', 'on antithyroid medication',
                     'sick', 'pregnant', 'thyroid surgery', 'I131 treatment', 'query hypothyroid',
                     'query hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary']

vector1 = data.loc[0, binary_attributes].astype(str)
vector2 = data.loc[1, binary_attributes].astype(str)

f11 = sum((vector1 == '1') & (vector2 == '1'))
f01 = sum((vector1 == '0') & (vector2 == '1'))
f10 = sum((vector1 == '1') & (vector2 == '0'))
f00 = sum((vector1 == '0') & (vector2 == '0'))

if f01 + f10 + f11 != 0:
    jc = f11 / (f01 + f10 + f11)
else:
    jc = 0.0

if f00 + f01 + f10 + f11 != 0:
    smc = (f11 + f00) / (f00 + f01 + f10 + f11)
else:
    smc = 0.0

print("Jaccard Coefficient (JC):", jc)
print("Simple Matching Coefficient (SMC):", smc)


# In[13]:


#A5


# In[14]:


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

file_name = r"19CSE305_LabData_Set3.1.xlsx"
worksheet_name = 'thyroid0387_UCI'
data = pd.read_excel(file_name, sheet_name=worksheet_name)

vector1 = data.iloc[0, 1:].apply(lambda x: float(x) if str(x).replace('.', '', 1).isdigit() else np.nan)
vector2 = data.iloc[1, 1:].apply(lambda x: float(x) if str(x).replace('.', '', 1).isdigit() else np.nan)

dot_product = np.dot(vector1, vector2)

magnitude_vector1 = np.linalg.norm(vector1)
magnitude_vector2 = np.linalg.norm(vector2)

cosine_similarity = dot_product / (magnitude_vector1 * magnitude_vector2)

print("Cosine Similarity:", cosine_similarity)


# In[15]:


#A6


# In[16]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity

file_name = "19CSE305_LabData_Set3.1.xlsx"
worksheet_name = "thyroid0387_UCI"
data = pd.read_excel(file_name, sheet_name=worksheet_name)

vectors = data.iloc[:20, 1:-1]

def jaccard_coefficient(vector1, vector2):
    intersection = np.logical_and(vector1, vector2)
    union = np.logical_or(vector1, vector2)
    return np.sum(intersection) / np.sum(union)

jc_matrix = np.zeros((20, 20))
smc_matrix = np.zeros((20, 20))
cosine_matrix = np.zeros((20, 20))

for i in range(20):
    for j in range(20):
        vector1 = vectors.iloc[i].astype(bool)
        vector2 = vectors.iloc[j].astype(bool)
        jc_matrix[i, j] = jaccard_coefficient(vector1, vector2)
        smc_matrix[i, j] = jaccard_score(vector1, vector2, average='binary')
        cosine_matrix[i, j] = cosine_similarity([vector1], [vector2])[0, 0]

plt.figure(figsize=(10, 8))
sns.heatmap(jc_matrix, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=False, yticklabels=False)
plt.title("Jaccard Coefficient Heatmap")
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(smc_matrix, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=False, yticklabels=False)
plt.title("Simple Matching Coefficient Heatmap")
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(cosine_matrix, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=False, yticklabels=False)
plt.title("Cosine Similarity Heatmap")
plt.show()


# In[ ]:




