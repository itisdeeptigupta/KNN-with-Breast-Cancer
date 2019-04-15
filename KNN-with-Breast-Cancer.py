
# coding: utf-8

# In[56]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score

from sklearn.neighbors import KNeighborsClassifier

import math


# In[10]:


dataset = pd.read_csv("C:\\MyRWork\\Data\\UCI-Breast-Cancer-Wisconsin\\breast-cancer-wisconsin.data", header=["Sample_code_number"
,"Clump_Thickness"
,"Uniformity_of_Cell_Size"
,"Uniformity_of_Cell_Shape"
,"Marginal_Adhesion"
,"Single_Epithelial_Cell_Size"
,"Bare_Nuclei"
,"Bland_Chromatin"
,"Normal_Nucleoli"
,"Mitoses"
,"Class"])


# In[22]:


dataset.head()
print(dataset.info)
print(dataset.dtypes)


# In[13]:


# Does it contain any missing value

dataset[dataset.isnull().any(axis=1)].head()


# In[17]:


# Does it contain any zero values

dataset[dataset.isin([0]).any(axis=1)].head()


# In[28]:


# Does it contain any special symbols

dataset[dataset.isin(['?']).any(axis=1)].head()


# In[30]:


Spl_Char_Not_Accepted = ["Sample_code_number"
,"Clump_Thickness"
,"Uniformity_of_Cell_Size"
,"Uniformity_of_Cell_Shape"
,"Marginal_Adhesion"
,"Single_Epithelial_Cell_Size"
,"Bare_Nuclei"
,"Bland_Chromatin"
,"Normal_Nucleoli"
,"Mitoses"
,"Class"]

for column in Spl_Char_Not_Accepted:
    dataset[column] = dataset[column].replace('?', np.NaN)


# In[31]:


dataset['Bare_Nuclei'] = dataset[['Bare_Nuclei']].apply(pd.to_numeric)


# In[32]:


for column in Spl_Char_Not_Accepted:
    mean = int(dataset[column].mean(skipna = True))
    dataset[column] = dataset[column].replace(np.NaN, mean)


# In[34]:


dataset.iloc[[23,40,139,145,158]]


# In[36]:


# Split the dataset

X = dataset.iloc[:,0:10]
y = dataset.iloc[:,10]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.2)

# Feature Scaling
Sc_X = StandardScaler()
X_train = Sc_X.fit_transform(X_train)
X_test = Sc_X.fit_transform(X_test)


# In[45]:


# decide for n_neighbors = k, because the dataset has 699 rows

math.sqrt(len(y_test))

# take 11 or 13 to start with classifier,odd values will make it easy to pick the group


# In[44]:


# Define the model : Init Knn

Classifier = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')
Classifier.fit(X_train, y_train)


# In[49]:


y_pred = Classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
cm


# In[57]:


print("accuracy_score: ", accuracy_score(y_test, y_pred))
print("f1_score: ", f1_score(y_test, y_pred, average="macro"))
print("precision_score: ", precision_score(y_test, y_pred, average="macro"))

