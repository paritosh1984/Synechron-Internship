import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras.layers import ConvLSTM2D

# Importing Data

d = pd.read_csv('Sample_Data.csv')

# Renaming column names

d.rename(columns = {'CCY Pair' : 'ccypair','CCY Amount' : 'ccyamt'}, inplace = True)

d = d.drop(columns = ["Operator","Unnamed: 1","CPTY","Source","CCY","count"])

d['ccypair'] = d['ccypair'].str.replace(' ','')

# Filtering the Values for FXFWD and INOPICS

fxfwd = []
for row in d.itertuples():
    if str(row[1]) == 'FXFWD' and str(row[2]) == 'INOPICS':
        fxfwd.append(row[:])         
fxfwd = pd.DataFrame(fxfwd, columns = ["Index","Product","Status","ccypair",'ccyamt'])
fxfwd = fxfwd.groupby(['ccypair'])['ccyamt'].sum().reset_index()
fxfwd.set_index('ccypair',inplace = True)
fxfwd.columns = [''] * len(fxfwd.columns)
fxfwd = fxfwd.transpose()
fxfwd = fxfwd.iloc[:,:1].reset_index()
fxfwd = fxfwd.drop(columns = ["index"])


"""# Simulating the Time Series
##number of point of the time series
nsample = 1024
## Simulate a simple sinusoidal function
x1 = np.linspace(0, 100, nsample)
y=np.sin(x1) + 13826"""

import statsmodels.api as sm
from statsmodels.tsa.arima_process import arma_generate_sample
np.random.seed(12345)

arparams = np.array([0.9, -0.9])
maparams = np.array([-0.5, 1])
ar = np.r_[1, -arparams]
ma = np.r_[1, maparams]

y = pd.DataFrame(arma_generate_sample(arparams,maparams, 499) + 13825.680000, columns = ["AUD/USD"])

fxfwd = pd.concat([fxfwd,y])
#plt.plot(fxfwd)

# Splitting the Dataset

train = fxfwd[:int(0.8*(len(fxfwd)))]
valid = fxfwd[int(0.8*(len(fxfwd))):]


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)
 
# define input sequence
raw_seq = list(train.iloc[:,:1].values)
# choose a number of time steps
n_steps = 6
n_features = 1
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# summarize the data
for i in range(len(X)):
	print(X[i], y[i])

X = X.reshape((X.shape[0], X.shape[1], n_features))

"""# Defining Model (Vanilla LSTM)
    
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Fit Model

model.fit(X, y, epochs=200, verbose=0)

# Prediction 
x_input = np.array(train.iloc[-6:,:1].values)
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat) [[13820.148]]
print(13821.07686246)"""

"""# Stacked LSTM

model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences = True,input_shape=(n_steps, n_features)))
model.add(LSTM(50,activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Fit Model

model.fit(X, y, epochs=200, verbose=0)

# Prediction
 
x_input = np.array(train.iloc[-6:,:1].values)
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat) [[13819.627]]
print(13821.07686246)

# Bidirectional LSTM

model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'),input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Fit Model

model.fit(X, y, epochs=200, verbose=0)

# Prediction
 
x_input = np.array(train.iloc[-6:,:1].values)
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat) [[13820.256]]
print(13821.07686246)

# CNN LSTM

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)
 
# define input sequence
raw_seq = list(train.iloc[:,:1].values)
# choose a number of time steps
n_steps = 6
# split into samples
X, y = split_sequence(raw_seq, n_steps)
#print(X.shape[0])
# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
n_features = 1
n_seq = 3
n_steps = 2
X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
#print(X)
# define model
model = Sequential()
model.add(TimeDistributed(Convolution1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=500, verbose=0)
# demonstrate prediction
x_input =  np.array(train.iloc[-6:,:1].values)
x_input = x_input.reshape((1, n_seq, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(x_input)
print(yhat)[[13821.233]]
print(train["AUD/USD"].iloc[-1]) # 13822.455313386075


# ConvLSTM

# Splitting Univariate sequence into samples
def split_sequence(seq,n_steps):
    X,y = list(), list()
    for i in range(len(seq)):
        end_ix = i + n_steps
        if end_ix > len(seq)-1:
            break
        seq_x, seq_y = seq[i:end_ix], seq[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# define input sequence
raw_seq = list(train.iloc[:,:1].values)
# choose a number of time steps
n_steps = 6
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
n_features = 1
n_seq = 3
n_steps = 2
X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))
# define model
model = Sequential()
model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
model.add(Flatten())
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=500, verbose=0)
# demonstrate prediction
x_input = np.array(train.iloc[-6:,:1].values)
x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)[[13821.135]]
print(train["AUD/USD"].iloc[-1]) # 13822.455313386075"""



















