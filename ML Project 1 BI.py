#!/usr/bin/env python
# coding: utf-8

# # ML Project - Bank Marketing Prediction

# In[1]:


# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data = pd.read_csv("C:/Users/Dell/Desktop/bank-marketing.csv")


# In[3]:


data.info()


# Hence dataset does not contain any missing value.

# In[4]:


data.describe()

# 1. Describe the pdays column, make note of the mean, median and minimum values. Anything fishy in the value
# In[5]:


data.pdays.describe()


# If we purely look at numerical summary ie mean and standard deviation, we can't see that lot of values is -1. We can see that 75% values of pdays are -1. So -1 has special meaning over here ie previous campaign was made to them or not. So in our case if we want to make decision on customer who did have campaign previously, then we must exclude all the cases of -1. So, by doing this we can get to customer who had previously campaign.
# 2. Describe the pdays column again, this time limiting yourself to the relevant values of pdays. How different are the mean and the median values?
# pdays uses -1 as indicator and not value. Hence treat these value as missing
# 
# Ignore these values in our average/median/state calculations.
# Keep i NaN
# Wherever pdays is -1, replace with NaN

# In[6]:


data1=data.copy()


# In[7]:


data1.drop(data1[data1['pdays'] < 0].index, inplace = True) 


# In[8]:


data1.pdays.describe()


# This time mean and median has changed significantly because we have removed the case where pdays value is -1 ie we have removed the customer that were not contacted previously for campaign.
# 3. Plot a horizontal bar graph with the median values of balance for each education level value. Which group has the highest median?
# In[9]:


data1.groupby(['education'])['balance'].median().plot.barh()


# Thus, we can conclude from graph that customer with tertiary level of education has highest median value for balance.
# 4. Make a box plot for pdays. Do you see any outliers?
# In[10]:


data1.pdays.plot.box()
plt.show()


# Yes, from the above box plot we can see that there are outliers present in pdays.

# # The final goal is to make a predictive model to predict if the customer will respond positively to the campaign or not. The target variable is “response”. So performing bi-variate analysis to identify the features that are directly associated with the target variable.

# # Bi- variate Analysis
# 5.Converting the response variable to a convenient form
# In[11]:


data1.response.value_counts(normalize=True)


# In[12]:


data1.replace({'response': {"yes": 1,'no':0}},inplace=True)


# In[13]:


data1.response.value_counts()

# 6. Make suitable plots for associations with numerical features and categorical features’
# In[14]:


# here we are seperating object and numerical data types 
obj_col = []
num_col = []
for col in data1.columns:
    if data1[col].dtype=='O':
        obj_col.append(col)
    else:
        num_col.append(col)


# In[15]:


print("Object data type features ",obj_col)
print("Numerical data type features ",num_col)


# In[16]:


from numpy import median
for col in obj_col[1:]:
    plt.figure(figsize=(8,6))
    sns.violinplot(data1[col],data1["response"])
    plt.title("Response vs "+col,fontsize=15)
    plt.xlabel(col,fontsize=10)
    plt.ylabel("Response",fontsize=10)
    plt.show()
#sns.despine()
# violin plots give best of both worlds 
# it gives boxplot and distribution of data like whether the data is skewed or not.
# if normally distributed then it's the best you can get.
# you can also use barplots in this case.


# In[17]:


for col in num_col[:-1]:
    plt.figure(figsize=(10,8))
    sns.jointplot(x = data1[col],y = data1["response"],kind='reg')
    plt.xlabel(col,fontsize = 15)
    plt.ylabel("Response",fontsize = 15)
    plt.grid()
    plt.show()


# In[18]:


plt.figure(figsize=(8,6))
sns.heatmap(data1.corr(),annot=True,cmap='RdBu_r')
plt.title("Correlation Of Each Numerical Features")
plt.show()


# we can see that duration variable is highly correlated with response variable 'Response Flag' . Whereas pdays variable is not highly correlated with response variable 'Response Flag'.

# # Are the features about the previous campaign data useful

# As we can observe in the above corelation matrix, previous campaign data is not much corelated and have only 0.0086 almost close to 0, so previous data cannot be used to predict.

# # Are pdays and poutcome associated with target

# In[19]:


pd.crosstab(data1['pdays'],data1['poutcome'],values=data1['response'],aggfunc='count',margins= True,normalize=True)


# Observation : pdays and poutcomes are associated with each other.
Label Encoding of Categorical Variables.
# In[20]:


from sklearn.preprocessing import LabelEncoder


# In[21]:


data2 = data1[obj_col].apply(LabelEncoder().fit_transform)


# In[22]:


data2.head()


# In[23]:


data3 = data2.join(data1[num_col])


# In[24]:


data3.head()


# In[25]:


data3.corr()


# # Train test Split

# In[26]:


X = data3.drop('response',axis=1)
y = data3['response']


# # Predictive model 1: Logistic regression
# 1. Before we build a model, lets select top features using RFE, so that model predict better

# In[27]:


from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[28]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)


# In[29]:


X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[30]:


lm = LogisticRegression()
lm.fit(X_train,y_train)


# In[31]:


rfe = RFE(lm,10)
rfe = rfe.fit(X_train,y_train)
rfe_ = X_train.columns[rfe.support_]
rfe_


# In[32]:


def checkVIF(X):
    vif = pd.DataFrame()
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'],2)
    vif = vif.sort_values(by = 'VIF',ascending = False)
    return vif


# In[33]:


X_train_new = X_train[rfe_]


# In[34]:


checkVIF(X_train_new)


# As Month of dec has highest VIF value,we will remove that feature
# 
# -- so we will consider the rest of the features for building our model

# In[35]:


data4 = data3[['job', 'marital', 'education', 'targeted', 'default', 'housing', 'loan',
       'contact', 'poutcome', 'campaign']]


# In[36]:


X_train_new,X_test_new,y_train,y_test = train_test_split(data4,y,test_size=0.3,random_state=0)


# In[37]:


z = lm.fit(X_train_new,y_train)
z


# # Estimate the model performance using k fold cross validation

# In[38]:


auc = [X_train,X_train_new]
models = []
models.append(('LogisticsRegression',LogisticRegression()))
for i in auc:
    kfold = KFold(n_splits=10)
    cv_results = cross_val_score(LogisticRegression(),i,y_train,cv=kfold)
    msg ='%s: %f (%f)' % (LogisticRegression, cv_results.mean(),cv_results.std())
    print(msg)


# By using the features we got from VIF and KFold ,we have got an accuracy of 81%.

# In[39]:


y_pred = z.predict(X_test_new)
y_pred


# In[40]:


X_test_new.shape


# In[41]:


y_pred.shape


# In[42]:


from sklearn.metrics import classification_report


# # Precision, Recall, Accuracy of your model

# In[43]:


print(classification_report(y_test,y_pred))


# In[44]:


importance = lm.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))

plt.bar([x for x in range(len(importance))], importance)
plt.show()


# # Most important from your model
# - poutcome,JOB,EDUCATION

# # Predictive model 2: Random Forest

# In[57]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# In[46]:


rfc = RandomForestClassifier(n_estimators=30, max_depth=30)
rfc.fit(X_train,y_train)


# In[47]:


y_pred = rfc.predict(X_test)
y_pred


# In[48]:


print(classification_report(y_test,y_pred))


# # Estimate the model performance using k fold cross validation

# In[49]:


p = [X_train,X_train_new]
for i in p:
    kfold = KFold(n_splits=10)
    cv_results = cross_val_score(RandomForestClassifier(),i,y_train,cv=kfold)
    msg ='%s: %f (%f)' % (RandomForestClassifier, cv_results.mean(),cv_results.std())
    print(msg)


# # As we can observe above
# - CV score(for all features) :0.84
# - CV score (for selected features) : 0.79

# In[50]:


model_new = RandomForestClassifier(n_estimators =45,max_depth=10)
model_new.fit(X_train_new,y_train)


# In[51]:


y1_pred = model_new.predict(X_test_new)


# In[52]:


print('For selected features')
print(accuracy_score(y_test,y1_pred))


# In[53]:


print(classification_report(y_test,y1_pred))


# In[54]:


feature_scores = pd.Series(model_new.feature_importances_, index=X_train_new.columns).sort_values(ascending=False)
feature_scores


# In[55]:


f, ax = plt.subplots(figsize=(10, 8))
ax = sns.barplot(x=feature_scores, y=feature_scores.index)
ax.set_title("Visualize feature scores of the features")
ax.set_yticklabels(feature_scores.index)
ax.set_xlabel("Feature importance score")
ax.set_ylabel("Features")
plt.show()


# # Evaluate both models on the test set

# In[58]:


print(" LR Test Accuracy:",metrics.accuracy_score(y_test,y_pred))
print(" RF Test Accuracy:",metrics.accuracy_score(y_test,y1_pred))


# From the above result Logestic Model has better accuracy

# In[59]:


from sklearn import model_selection
seed=0
models = []
models.append(('LR', LogisticRegression()))
models.append(('RF', RandomForestClassifier()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10,shuffle=True, random_state=0)
    cv_results = model_selection.cross_val_score(model, i, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# # Which metric did you choose and why?
# we used classification_report metrics such as, Precision score,accuracy score,recall score and Cross val score etc.

# # Which model has better performance on the test set?
# Logistic has got better accuracy score campared to random foreset, hence we can say that it has better performance is an important model as it results in high AUC score

# # Compare the feature importance from the different models – do they agree? Are the top features similar in both models?
# According to LogisticRegression Model model, poutcome ,job and education are three most important features and poutcome,housing and job are three most important features In RandomFrest Model model. on comparing both the model, we can say poutcome and job are the similar top feature. education is second most important feature in LosgiticRegression model while in Random Forest, is least important.

# In[ ]:




