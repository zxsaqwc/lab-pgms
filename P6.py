#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

c1,c2,c3,c4 = np.loadtxt('data.csv',unpack=True,delimiter = ',')
x= np.column_stack((c1,c3))
y= c4
#Create NaiveBayes Classifier
clf = GaussianNB()
#fit the mode
clf.fit(x,y)
#make predictions
predictions = clf.predict(x)

#calculate accuracy
print(accuracy_score(y,predictions)*100)


# In[ ]:


1.0, 1.0, 1.0, 1.0
1.0, 1.0, 1.0, 2.0
2.0, 1.0, 1.0, 2.0
3.0, 2.0, 1.0, 1.0
3.0, 3.0, 2.0, 1.0

