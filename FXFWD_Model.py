
import pandas as pd
import numpy as np
from random import randint
import matplotlib.pyplot as plt
import statsmodels.api as sm
import random
import seaborn as sns

df = pd.read_csv('Sample_Data.csv')

# Renaming column names

df.rename(columns = {'CCY Pair' : 'ccypair','CCY Amount' : 'ccyamt'}, inplace = True)

df = df.drop(columns = ["Operator","Unnamed: 1","CPTY","Source","CCY","count"])

df['ccypair'] = df['ccypair'].str.replace(' ','')

# Filtering the Values for FXFWD and INOPICS

fxfwd = []
for row in df.itertuples():
    if str(row[1]) == 'FXFWD' and str(row[2]) == 'INOPICS':
        fxfwd.append(row[:])         
fxfwd = pd.DataFrame(fxfwd, columns = ["Index","Product","Status","ccypair",'ccyamt'])
fxfwd = fxfwd.groupby(['ccypair'])['ccyamt'].sum().reset_index()
fxfwd.set_index('ccypair',inplace = True)
fxfwd = fxfwd.transpose()

# Adding random values for all currency pair

for i in range(0,500):
     row_dict = {}
     for key,value in fxfwd.items():
         value = fxfwd[key][0]
         row_dict[key] = random.randint(0,int(value))
     fxfwd = fxfwd.append(row_dict,ignore_index = True)
     
"""# Taking care of Missing Values
        
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = np.nan, strategy = 'mean', axis = 0)
imputer = imputer.fit(fxfwd[1:])
fxfwd[1:] = imputer.transform(fxfwd[1:])
#fxfwd.to_csv(r'/Users/paritoshkalla/Desktop/Machine Learning/Fw__Discussion_on_AI_ML__Visual_Research/Follow_up_discussion_on_AIML_scenarios06192019_/fxfwd.csv')"""


corr = fxfwd.corr()
corr.style.background_gradient(cmap='coolwarm')



corrmat = fxfwd.corr() 
f, ax = plt.subplots(figsize =(9, 8)) 
sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1)

"""# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_fwd = StandardScaler()
fxfwd[:] = sc_fwd.fit_transform(fxfwd[:])"""


# Plots for Checking the Stationarity

for key,value in fxfwd.iteritems():
    fig = plt.Figure(figsize = (10,70))
    layout = (2,2)
    ax1 = plt.subplot2grid(layout,(0,0))
    fxfwd[[key]].plot(ax = ax1)    
    ax2 = plt.subplot2grid(layout,(0,1))
    fxfwd[[key]].hist(ax = ax2, bins = 15)
    ax3 = plt.subplot2grid(layout,(1,0))
    sm.graphics.tsa.plot_acf(fxfwd[key], lags=15,ax = ax3)
    ax4 = plt.subplot2grid(layout,(1,1))
    sm.graphics.tsa.plot_pacf(fxfwd[key], lags=15,ax = ax4)
    plt.show()

# Calculating the Mean and Variance for each currency pair to check for the Stationarity

for key,value in fxfwd.iteritems():
    X1,X2 = value[0:10],value[10:]
    m1 = X1.mean()
    m2 = X2.mean()
    var1 = X1.var()
    var2 = X2.var()
    print("Mean of two groups for currency pair {} is {} and {}".format(key,m1,m2))
    print("Variance of two groups for currency pair {} is {} and {}".format(key,var1,var2))
      

# Test for Stationarity (Johansen Cointegration Test)

from statsmodels.tsa.vector_ar.vecm import coint_johansen
print(coint_johansen(fxfwd,-1,1).eig)


# Creating the train and validation set

train = fxfwd[:int(0.8*(len(fxfwd)))]
valid = fxfwd[int(0.8*(len(fxfwd))):]


# Fit the model (Vector Autoregressive Model)

from statsmodels.tsa.vector_ar.var_model import VAR

model = VAR(endog=train.astype(float))
model_fit = model.fit()

# Make prediction on validation

prediction = model_fit.forecast(model_fit.y, steps=len(valid))
valid = pd.DataFrame(valid, columns = fxfwd.columns)
#valid.to_csv(r'/Users/paritoshkalla/Desktop/Machine Learning/Fw__Discussion_on_AI_ML__Visual_Research/Follow_up_discussion_on_AIML_scenarios06192019_/fxfwd_test.csv')

"""# Forecasting for next 7 days

pred7 = model_fit.forecast(model_fit.y, steps=7)
pred7 = pd.DataFrame(pred7, columns = fxfwd.columns)
pred7.to_csv(r'/Users/paritoshkalla/Desktop/Machine Learning/Fw__Discussion_on_AI_ML__Visual_Research/Follow_up_discussion_on_AIML_scenarios06192019_/pred7.csv')


# Converting predictions to dataframe
pred = pd.DataFrame(index=range(0,len(prediction)),columns=fxfwd.columns)
for j in range(0,10):
    for i in range(0, len(prediction)):
       pred.iloc[i][j] = prediction[i][j]

# Check RMSE
import math
from sklearn.metrics import mean_squared_error 
for i in fxfwd.columns:
    print('RMSE value for', i, 'is : ', math.sqrt(mean_squared_error(pred[i], valid[i])))

# Make Final Predictions

model = VAR(endog=fxfwd)
model_fit = model.fit()
yhat = model_fit.forecast(model_fit.y, steps=1)
print(yhat)

valid = pd.DataFrame(valid, columns = fxfwd.columns)
#valid.to_csv(r'/Users/paritoshkalla/Desktop/Machine Learning/Fw__Discussion_on_AI_ML__Visual_Research/Follow_up_discussion_on_AIML_scenarios06192019_/fxfwd_test.csv')"""























































































'''

for value in data.itertuples():
    if value[1] == 'FXSPOT' and value[3] == 'INOPICS':
        fxspot = data.groupby(['ccypair'])['ccyamt'].sum().reset_index()
    elif value[1] == 'FXFWD' and value[3] == 'INOPICS':
        fxfwd = data.groupby(['ccypair'])['ccyamt'].sum().reset_index()
        
'''

'''

# Grouping by the rows 

new_df = df.groupby(['Product','Status','ccypair'])['ccyamt'].sum().reset_index()


fxfwd = new_df[new_df['Product'] == 'FXFWD']
fxspot = new_df[new_df['Product'] == 'FXSPOT']

fxfwd = fxfwd[fxfwd.apply(lambda x:x['Status'] == 'INOPICS', axis = 1)]
fxspot = fxspot[fxspot.apply(lambda x:x['Status'] == 'INOPICS', axis = 1)]

# Removing spaces

fxspot['ccypair'] = fxspot['ccypair'].str.replace(' ','')

# Grouping the rows

fxspot = fxspot.groupby(['ccypair'])['ccyamt'].sum().reset_index()
fxfwd = fxfwd.groupby(['ccypair'])['ccyamt'].sum().reset_index()

'''

'''
# Transpose

fxspot = fxspot.transpose()

'''

'''
# Adding random values
 
from numpy.random import randint

r = np.random.randint(45000, size = (10000,20))

r = pd.DataFrame(r)

frames = [fxspot,r]

newfxspot = pd.concat(frames)

'''
