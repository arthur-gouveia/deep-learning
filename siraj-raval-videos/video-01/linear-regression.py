
# coding: utf-8

# # Predicting body weight based on brain weight using linear regresssion

# In[2]:

import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt


# In[3]:

dataframe = pd.read_fwf('brain_body.txt')


# In[29]:

X = dataframe[['Brain']]
y = dataframe[['Body']]
model = linear_model.LinearRegression()
model.fit(X, y)


# In[31]:

plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.show()


# In[33]:

model.score(X, y)

