#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression  # Example model
from sklearn.datasets import make_classification       # Example dataset
import numpy as np


# In[2]:


X , y =make_classification(n_samples=1000, n_features=20, random_state=42)


# In[19]:


X.shape


# In[4]:


kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression()

scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')



# In[5]:


print("K-Fold Cross-Validation Scores:", scores)
print("Mean Accuracy:", np.mean(scores))
print("Standard Deviation of Accuracy:", np.std(scores))


# In[26]:


## manual implementaion of Kfold

def manual_kflod(X, y, k =5):
    fold_size = len(X)//5
    scores = []
    for i in range(k):
        start = i * fold_size
        end = (i+1) * fold_size
        X_val = X[start:end]
        y_val = y[start:end]
        #print(X.shape, y.shape)
        print(X_val.shape, y_val.shape)

        X_train = np.concatenate([X[:start], X[:end]])
        y_train = np.concatenate([y[:start], y[:end]])

        model = LogisticRegression()
        #print(X_train.shape, y_train.shape)
        model.fit(X_train, y_train)
        score = model.score(X_val,y_val)

        scores.append(score)
    
    return scores


# In[27]:


scores_manual = manual_kflod(X, y)
print("Manual score : ", scores_manual)


# In[ ]:




