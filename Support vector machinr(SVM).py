#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd


# In[3]:


from sklearn.datasets import load_iris
iris=load_iris()


# In[4]:


dir(iris)


# In[5]:


iris.feature_names


# In[9]:


df=pd.DataFrame(iris.data,columns=iris.feature_names)


# In[10]:


df.head()


# In[11]:


iris.data


# In[12]:


df['target']=iris.target


# In[13]:


df.head()


# In[14]:


iris.target_names


# In[16]:


df[df.target==2].head()


# In[17]:


df['flower_name']=df.target.apply(lambda x:iris.target_names[x])


# In[18]:


df.head()


# In[19]:


from matplotlib import pyplot as plt


# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


df0=df[df.target==0]


# In[22]:


df1=df[df.target==1]
df2=df[df.target==2]


# In[23]:


df.head()


# In[26]:


plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')

plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='green')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='blue')


# In[27]:


plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')

plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color='green')
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='blue')


# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


x=df.drop(['target','flower_name'],axis='columns')
x.head()


# In[30]:


y=df.target


# In[31]:


y


# In[32]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[33]:


len(x_train)


# In[34]:


len(x_test)


# In[35]:


from sklearn.svm import SVC


# In[36]:


model=SVC()


# In[37]:


model.fit(x_train,y_train)


# In[38]:


model.score(x_test,y_test)


# In[ ]:




