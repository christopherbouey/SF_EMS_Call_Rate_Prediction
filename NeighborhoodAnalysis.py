#!/usr/bin/env python
# coding: utf-8

# In[61]:


import pandas as pd
import numpy as np
import random
from datetime import timedelta
import seaborn as sns


# In[62]:


import boto3
import matplotlib.pyplot as plt


# In[63]:


s3 = boto3.client('s3')


# In[64]:


file = s3.get_object(Bucket='sf-ems-analysis', Key='EMS_no_dup.csv')


# In[65]:


def pd_dttm_import(want_cols):
    '''Parameters: takes in list of column names wanted that will be
    used in pandas .read_csv method
    
    Results: Outputs the indices of the columns names that are date
    values and uses resulting list of indices as parse_dates parameter
    in pandas .read_csv method'''
    dt_ind = []
    for i in want_cols:
        if 'dttm' in i.lower() or 'date' in i.lower():
            dt_ind.append(want_cols.index(i))
    return dt_ind

#import only certain columns from csv
col_of_interest = ['Call Number', 'Call Type',
       'Received DtTm', 'Call Final Disposition', 
       'Final Priority',
       'Call Type Group',
       'Neighborhooods - Analysis Boundaries']


# In[66]:


df = pd.read_csv(file['Body'], usecols=col_of_interest, parse_dates=pd_dttm_import(col_of_interest))


# In[67]:


#set call time stamp to index
df.set_index(keys=df['Received DtTm'], inplace = True)


# In[68]:


#drop call time stamp columns
df.drop(columns=['Received DtTm'], inplace = True)


# In[69]:


df.head()


# In[70]:


#Investigate usefulness of column
df['Call Final Disposition'].value_counts()


# In[71]:


#drop column - does not seem useful
df.drop(columns=['Call Final Disposition'], inplace = True)


# In[72]:


#Investigate usefulness of column
df['Final Priority'].value_counts()


# In[73]:


#Investigate usefulness of column
df['Call Type Group'].value_counts()


# In[74]:


#drop column - repetitive
df.drop(columns=['Call Type Group'], inplace=True)


# In[75]:


#shorten to two year period
df_dated = df[(df.index > '2016-08-16') & (df.index < '2018-08-23')]


# In[76]:


#identify 3 most common call types
top3 = df_dated['Call Type'].value_counts()[0:3]


# In[78]:


#make all call types 'other' except top three
df_dated['Call Type'] = [x if x in top3 else 'Other' for x in df_dated['Call Type']]
df_dated['Call Type'].value_counts()


# In[80]:


#create dummy variables out of features
df_dated_dummies = pd.get_dummies(df_dated, columns=['Call Type','Final Priority','Neighborhooods - Analysis Boundaries'])


# In[88]:


# set index to datetime object
df_dated_dummies.index = pd.to_datetime(df_dated_dummies.index)


# In[95]:


#take count of target variable (number of calls per 3H period)
agg = df_dated_dummies['Call Number'].resample('3H').count()


# In[94]:


#take sum of dummy variables to have as feature
agg_sum = df_dated_dummies.iloc[:,1:].resample('3H').sum()


# In[100]:


#merge the two
agg_tot = pd.merge(agg, agg_sum, on='Received DtTm')


# ## Explore by Neighborhood

# In[144]:


#divide dataframe into top 10 busiest neighborhoods
df = df[df.index > '2000-04-14 00:00:00']
neighs = {}
for i, j in enumerate(df['Neighborhooods - Analysis Boundaries'].value_counts().index[:10]):
    neighs[i] = j

frames = {}
for i, j in enumerate(df['Neighborhooods - Analysis Boundaries'].value_counts().index[:10]):
    dfi = df[df['Neighborhooods - Analysis Boundaries'] == j]
    dfi.index = pd.to_datetime(dfi.index)
    frames[i] = dfi


# ### Seasonal Decomposition of Data by Neighborhood

# In[214]:


from scipy import signal
from scipy import stats
plt.style.use('seaborn-darkgrid')
import statsmodels.api as sm
for i, j in zip(neighs.values(), frames.values()):
    fra = j['Call Number'].resample('H').count()
    series_decomposition = sm.tsa.seasonal_decompose(fra, freq=24)
    idx = (fra.index.get_loc('2000-04-14 10:00:00'))
    series_decomposition.seasonal[idx:idx+24].plot(figsize=(15,7))
plt.legend(neighs.values(), prop={'size': 12})
plt.title('Daily Seasonality by Neighborhood')
plt.ylabel('Variation')
plt.xlabel('Time of Day')
plt.rcParams.update({'font.size': 14})
plt.savefig('Daily_Neigh.png')


# In[213]:


from scipy import signal
from scipy import stats
plt.style.use('seaborn-darkgrid')
import statsmodels.api as sm
for i, j in zip(neighs.values(), frames.values()):
    fra = j['Call Number'].resample('D').count()
    series_decomposition = sm.tsa.seasonal_decompose(fra, freq=7)
    idx = (fra.index.get_loc('2000-04-17'))
    series_decomposition.seasonal[idx:idx+7].plot(figsize=(15,7))
plt.legend(neighs.values(), prop={'size': 10})
plt.title('Weekly Seasonality by Neighborhood')
plt.ylabel('Variation')
plt.xlabel('Day of Week')
plt.rcParams.update({'font.size': 14})
plt.xticks(ticks= series_decomposition.seasonal[idx:idx+7].index, labels=['Mon','Tues','Wed','Thu','Fri','Sat','Sun'])
plt.savefig('Weekly_Neigh.png')


# In[189]:


vals=[]
plt.style.use('seaborn-darkgrid')
for i, j in zip(neighs.values(), frames.values()):
    vals.append (j['Call Number'].count())
#plt.figure(figsize=(12,6))    
plt.barh(np.arange(len(vals)), vals, )
plt.yticks(np.arange(len(vals)), neighs.values(), rotation=0)
plt.savefig('BarNeigh.png')


# In[212]:


plt.style.use('seaborn-darkgrid')
import statsmodels.api as sm
for i, j in zip(neighs.values(), frames.values()):
    fra = j['Call Number'].resample('M').count()
    series_decomposition = sm.tsa.seasonal_decompose(fra, freq=12)
    series_decomposition.seasonal[:12].plot(figsize=(15,7))
plt.legend(neighs.values(), prop={'size': 10})
plt.title('Monthly Seasonality by Neighborhood')
plt.ylabel('Variation')
plt.xlabel('Month')
plt.rcParams.update({'font.size': 14})
plt.savefig('Monthly_Neigh.png')


# In[211]:


fig, ax = plt.subplots(figsize=(15,7))
for i, j in zip(neighs.values(), frames.values()):
    fra = j['Call Number'].resample('Y').count()
    ax.plot(fra[1:-1])


plt.legend(neighs.values(), prop={'size': 10})
plt.title('Yearly Trend by Neighborhood')
plt.ylabel('Total EMS Calls')
plt.xlabel('Year')
plt.rcParams.update({'font.size': 16})

plt.savefig('Yearly_Neigh.png')


# In[166]:


plt.bar(np.arange(len(vals)), vals)


# In[167]:


np.arange(len(vals))


# ### Prophet Calculation by Neigh

# In[234]:


from math import sqrt
from sklearn.metrics import mean_squared_error
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


# In[235]:


prophs = {}
for i, j in enumerate(frames.values()):
    f = j['Call Number'].resample('3H').count()
    f = f[(f.index >= '2016-08-16 00:00:00') & (f.index <= '2018-08-23 00:00:00')]
    prophs[i] = f


# In[237]:


prophs[0]


# In[246]:


fig, ax = plt.subplots(10, figsize=(20,40))
dic = {}
for i, (axs,n) in enumerate(zip(ax.flatten(),prophs.values())):
    m=Prophet()
    k = pd.DataFrame(n)
    j = k.reset_index(level=0)
    j.columns = ['ds','y']
    m.fit(j.iloc[:-56])
    future=m.make_future_dataframe(periods=56, freq='3H')
    forecast = m.predict(future)
    yhat = forecast.iloc[-56:]
    actual = j.iloc[-56:]
    print(measure_rmse(actual['y'], yhat['yhat']))
    actual = j.iloc[-112:]
    yhat = forecast.iloc[-112:]
    axs.plot(actual.index, actual['y'])
    axs.plot(actual.index, yhat['yhat'])
    axs.tick_params(labelrotation=90)
    plt.legend(['Actual','Pred'])
    plt.xticks([x for x in range(25140,25252, 8)], ['Thu','Fri','Sat','Sun','Mon','Tues','Wed']*2)
    axs.set_title(f'{neighs[i]}')
plt.tight_layout()
plt.savefig('3H_Daily_Neigh.png')


# In[258]:


prophs = {}
for i, j in enumerate(frames.values()):
    f = j['Call Number'].resample('D').count()
    f = f[(f.index >= '2016-08-16') & (f.index <= '2018-08-23')]
    k = pd.DataFrame(f)
    j = k.reset_index(level=0)
    j.columns = ['ds','y']
    prophs[i] = j


# In[262]:


fig, ax = plt.subplots(10, figsize=(20,40))
dic = {}
for i, (axs,n) in enumerate(zip(ax.flatten(),prophs.values())):
    m=Prophet()
    m.fit(n.iloc[:-7])
    future=m.make_future_dataframe(periods=7, freq='D')
    forecast = m.predict(future)
    yhat = forecast.iloc[-7:]
    actual = n.iloc[-7:]
    print(measure_rmse(actual['y'], yhat['yhat']))
    actual = n.iloc[-14:]
    yhat = forecast.iloc[-14:]
    axs.plot(actual.index, actual['y'])
    axs.plot(actual.index, yhat['yhat'])
    axs.tick_params(labelrotation=90)
    plt.legend(['Actual','Pred'])
    axs.set_title(f'{neighs[i]}')
    axs.set_xticks([724,724+14], ['Thu','Fri','Sat','Sun','Mon','Tues','Wed']*2)
plt.tight_layout()
plt.savefig('Daily_Pred_Neigh.png')


# In[ ]:




