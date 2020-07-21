#!/usr/bin/env python
# coding: utf-8

# # by: Gaurav Lohkna

# # Loading libraries and Data

# In[113]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix ,accuracy_score
get_ipython().run_line_magic('matplotlib', 'inline')

#loading data
cancer = pd.read_csv("/home/gaurav/Downloads/datasets%2F180%2F408%2Fdata.csv")


# In[114]:


cancer.head()


# # Data wrangling

# In[52]:


sns.pairplot(cancer, hue = 'diagnosis', vars = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean'])


# In[19]:


plt.figure(figsize =(22,10))
sns.heatmap(cancer.corr(), annot = True)


# In[115]:


cancer = cancer.drop(['id'], axis = 1)
##removing id attribute because it was of no use for prediction
cancer.head()


# In[116]:


cancer = cancer.dropna(how='all', axis = 'columns')
##removing NULL value  attribute to reduce the data size and time 
cancer.head()


# In[117]:


cancer = cancer.replace({'diagnosis': {'M': 1, 'B': 0}})
#converting the string to a boolean for (M = malignant, B = benign)


# # Model training

# In[118]:


x = cancer


# In[119]:


y = cancer['diagnosis']
y.head()


# In[176]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.3,random_state = 112)


# In[177]:


svc_model = SVC(kernel = 'poly')  # kernel can be linear,poly,rbf
# choose linear if the dataset is linear else choose poly or rbf


# In[178]:


svc_model.fit(x_train,y_train)


# In[179]:


y_predict = svc_model.predict(x_test)


# # Prediction and Accuracy check

# In[180]:


conf_mat = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))


# In[181]:


matrix = pd.DataFrame(conf_mat, index=['is cancer', 'is healthy'], columns = ['predicted_cancer','predicted_healthy'])
matrix


# In[182]:


sns.heatmap(matrix, annot=True)


# In[183]:


print(classification_report(y_test,y_predict))


# In[184]:


accuracy_score(y_test,y_predict)*100

