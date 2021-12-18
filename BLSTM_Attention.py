# Remaining useful Life Prediction using Bidirectional LSTM and Attention Mechanism
# Script developed to attend ISE298 MS Project
# Professor Yupeng Wei
# Fall 2021

#%%
from keras_preprocessing import sequence
from matplotlib.colors import Normalize
from numpy.core.defchararray import index
from numpy.core.fromnumeric import reshape
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import os
from pandas.core import groupby
from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
import seaborn as sns
from keras.preprocessing.sequence import TimeseriesGenerator, pad_sequences
from keras.models import Sequential
from keras.layers import Bidirectional, Reshape
from keras.layers import Dense, LSTM, Dropout, Activation
from keras import metrics
from sklearn.metrics import mean_squared_error
from scipy import stats
import scipy.io
from tensorflow.python.ops.gen_logging_ops import histogram_summary


#%%
train_raw = pd.read_csv('/Users/raissaantunes/Python/Turbofan/Data/train_FD001.txt', sep=' ', header = None)
train_raw.head()
test_raw = pd.read_csv('/Users/raissaantunes/Python/Turbofan/Data/test_FD001.txt', sep=' ', header = None)
test_raw.head()
RUL_raw = pd.read_csv('/Users/raissaantunes/Python/Turbofan/Data/RUL_FD001.txt', sep=' ', header = None)
RUL_raw.head()


#%%
#train dataframe
train_raw = train_raw[[f for f in range(0,26)]]
train_raw.columns = ['Engine', 'Cycle', 'Operational Setting 1', 'Operational Setting 2', 'Operational Setting 3', 'T2', 'T24', 'T30', 
'T50', 'P2','P15','P30','Nf','Nc', 'Epr', 
'Ps30', 'Phi', 'NRf', 'NRc', 'BPR', 'farB', 
'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']
#test dataframe
test_raw = test_raw[[f for f in range(0,26)]]
test_raw.columns = ['Engine', 'Cycle', 'Operational Setting 1', 'Operational Setting 2', 'Operational Setting 3', 'T2', 'T24', 'T30', 
'T50', 'P2','P15','P30','Nf','Nc', 'Epr', 
'Ps30', 'Phi', 'NRf', 'NRc', 'BPR', 'farB', 
'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']



#%%  NORMALIZING TRAIN
#Create RUL feature calculate RUL
max_cycles_df = train_raw.groupby('Engine', sort=False)['Cycle'].max()
max_cycles_df = max_cycles_df.reset_index().rename(columns={'Cycle':'MaxCycle'})
max_cycles_df.head()
max_cycles_df.shape
#Merge back to original data FDOO1 and  
FDOO1_df = pd.merge(train_raw, max_cycles_df, how='inner', on='Engine')
FDOO1_df['RUL'] = FDOO1_df['MaxCycle'] - FDOO1_df['Cycle']
#calculate RUL as Y_train
len(FDOO1_df[FDOO1_df['RUL'] == 0])
y_train_df = FDOO1_df['RUL']
#selecionar sensors for normalization x_train
x_train = train_raw[['T24', 'T30','T50', 
'P30', 'Nf', 'Nc', 'Ps30', 'Phi', 
'NRf', 'NRc', 'BPR', 'htBleed', 'W31',
'W32']]
#normalizar selected sensors from train_raw and asign to x_train_norm
scaler = MinMaxScaler()
scaler.fit(x_train) #compute min-max for scaling
x_train_norm = scaler.transform(x_train) #scale features according to min-max range
x_train_norm



#%% NORMALIZING TEST
#selected sensors for x_test
x_test_df = test_raw[['T24', 'T30','T50', 
'P30', 'Nf', 'Nc', 'Ps30', 'Phi', 
'NRf', 'NRc', 'BPR', 'htBleed', 'W31',
'W32']]
#normalize X_test
scaler = MinMaxScaler()
scaler.fit(x_test_df) #compute min-max for scaling
x_test_norm = scaler.transform(x_test_df) #scale features according to min-max range
x_test_norm

#%%
#Define max Maxcycle among train and test datasets
max_cycles_test = test_raw.groupby('Engine', sort=False)['Cycle'].max()
max_cycles_test = max_cycles_test.reset_index().rename(columns={'Cycle':'MaxCycle'})
seq_length = (max(max(max_cycles_df.iloc[:,1]), max(max_cycles_test.iloc[:,1])))

#%%
#convert x_train_norm back to pd.Dataframe on x_train_df
train_df = pd.DataFrame(x_train_norm)
#Add FDOO1_df['Engine', 'RUL', ] to x_train_ndf to pad the whole train array
train_df['RUL'] = FDOO1_df['RUL'].values
train_df['Engine'] = FDOO1_df['Engine'].values



#%% PLOTS
#violin plot for RUL
plt.figure(figsize=(12,6))
ax = sns.violinplot(data=max_cycles_df['MaxCycle'])

#plot Max cycle histogram to check distribution
max_cycles_df.plot(x="Engine", y="MaxCycle", kind="hist")
plt.xlabel("Train RUL")
plt.ylabel("RUL Frequency")

#plot y_train
one_engine = []
for i,r in FDOO1_df.iterrows(): #interate in deatframe as (i=index/label, r=series) 
    rul = r['RUL']
    one_engine.append(rul)
    if rul == 0:
        plt.plot(one_engine)
        one_engine = []
plt.grid()
plt.xlabel("Train RUL")
plt.ylabel("RUL Frequency")

## Operational settings plot: OpSet vs. Cycle
FDOO1_df.plot.scatter(x='Cycle', y='Operational Setting 1', alpha=0.5)
FDOO1_df.plot.scatter(x='Cycle', y='Operational Setting 2', alpha=0.5)
FDOO1_df.plot.scatter(x='Cycle', y='Operational Setting 3', alpha=0.5)

## Sensors graphs plot to delete useless sensors
FDOO1_df.plot.scatter(x='Cycle', y='T2', alpha=0.5)
FDOO1_df.plot.scatter(x='Cycle', y='T24', alpha=0.5)
FDOO1_df.plot.scatter(x='Cycle', y='T30', alpha=0.5)
FDOO1_df.plot.scatter(x='Cycle', y='T50', alpha=0.5)
FDOO1_df.plot.scatter(x='Cycle', y='P2', alpha=0.5)
FDOO1_df.plot.scatter(x='Cycle', y='P15', alpha=0.5)
FDOO1_df.plot.scatter(x='Cycle', y='P30', alpha=0.5)
FDOO1_df.plot.scatter(x='Cycle', y='Nf', alpha=0.5)
FDOO1_df.plot.scatter(x='Cycle', y='Nc', alpha=0.5)
FDOO1_df.plot.scatter(x='Cycle', y='Epr', alpha=0.5)
FDOO1_df.plot.scatter(x='Cycle', y='Ps30', alpha=0.5)
FDOO1_df.plot.scatter(x='Cycle', y='Phi', alpha=0.5)
FDOO1_df.plot.scatter(x='Cycle', y='NRf', alpha=0.5)
FDOO1_df.plot.scatter(x='Cycle', y='NRc', alpha=0.5)
FDOO1_df.plot.scatter(x='Cycle', y='BPR', alpha=0.5)
FDOO1_df.plot.scatter(x='Cycle', y='farB', alpha=0.5)
FDOO1_df.plot.scatter(x='Cycle', y='htBleed', alpha=0.5)
FDOO1_df.plot.scatter(x='Cycle', y='Nf_dmd', alpha=0.5)
FDOO1_df.plot.scatter(x='Cycle', y='PCNfR_dmd', alpha=0.5)
FDOO1_df.plot.scatter(x='Cycle', y='W31', alpha=0.5)
FDOO1_df.plot.scatter(x='Cycle', y='W32', alpha=0.5)
plt.Axis.plot()



#%% PRE-PADDING TRAIN
#Pad each engine on x and y train and test sequences to the max MaxCycle
train_list = train_df.to_numpy().tolist()
#use for loop for each engine id in the array 
train_dict = {} #array of 100 padded engines
for sub_array in train_list: # for each sub-array in the train_array get engine id
    eng_id = sub_array.pop()
    if eng_id not in train_dict:
        train_dict[eng_id] = []
    train_dict[eng_id].append(sub_array)

#%%
#padding each engine in the dictionary with pre 0
#iterating through both key and values in the dict
for unique_id, sequence in train_dict.items():
    # print(unique_id,sequence)
    engine_t = np.transpose(train_dict[unique_id])
    engine_pad = pad_sequences(engine_t, maxlen=362, padding='pre', dtype='float32')
    engine_t_back = np.transpose(engine_pad)
    train_dict[unique_id] = engine_t_back

#%%
#PREDICT FOR EACH ENGINE SEPARATELY:
engine1 = pd.DataFrame(train_dict[1])
engine1
y_train_e1 = engine1.iloc[:,14]
y_train_e1

x_train_e1 = engine1.iloc[:, :14]
x_train_e1
#Convert Dictionary to list and np.array to be used on hte Time Series Generator

#%% PRE-PADDING TEST
y_test_df = RUL_raw[0]
y_test_df.columns = ['RUL']
y_test_df
y_test_df.describe()
y_test_df = pd.DataFrame(y_test_df)
y_test_df['Engine'] = [x for x in range(1,101)]
y_test_df.columns= ['RUL', 'Engine']

#take max from each engine by grouping engines by their max cycle
#group Engine by their maxcycle

max_cycles_test = test_raw.groupby('Engine', sort=False)['Cycle'].max()
max_cycles_test = max_cycles_test.reset_index().rename(columns={'Cycle':'MaxCycle'})
max_cycles_test.head()
max_cycles_test.shape

y_test_df["Total_life"] = y_test_df["RUL"] + max_cycles_test["MaxCycle"]
y_test_merged = pd.merge(test_raw, y_test_df, how='left' , on= 'Engine')
y_test_merged["RUL"] = y_test_merged["Total_life"] - y_test_merged["Cycle"]
y_test_merged
y_test = y_test_merged["RUL"]

#%%
#add whole test set to pad
#convert x_test_norm back to pd.Dataframe on test_df
test_df = pd.DataFrame(x_test_norm)
#Add FDOO1_df['Engine', 'RUL', ] to x_train_ndf to pad the whole train array
test_df['RUL'] = y_test_merged['RUL'].values
test_df['Engine'] = y_test_merged['Engine'].values



#%% PRE-PADDING TEST
test_list = test_df.to_numpy().tolist()
#use for loop for each engine id in the array 
test_dict = {} #array of 100 padded engines
for test_sub_array in test_list: # for each sub-array in the train_array get engine id
    test_eng_id = test_sub_array.pop()
    if test_eng_id not in test_dict:
        test_dict[test_eng_id] = []
    test_dict[test_eng_id].append(test_sub_array)

#%%
#padding each engine in the dictionary with pre 0
#iterating through both key and values in the dict
for test_unique_id, test_sequence in train_dict.items():
    # print(unique_id,sequence)
    test_engine_t = np.transpose(test_dict[test_unique_id])
    test_engine_pad = pad_sequences(test_engine_t, maxlen=362, padding='pre', dtype='float32')
    test_engine_t_back = np.transpose(test_engine_pad)
    test_dict[test_unique_id] = test_engine_t_back



#%%
    #PREDICT FOR EACH ENGINE SEPARATELY:
engine1_test = pd.DataFrame(test_dict[1])
engine1_test
y_test_e1 = engine1_test.iloc[:,14]
y_test_e1
x_test_e1 = engine1_test.iloc[:, :14]
x_test_e1

plt.plot(x_train_e1)
plt.plot(y_train_e1)
plt.plot(x_test_e1)
plt.plot(y_test_e1)



#%%
#Global parameters
win_length = 20
num_features = 14

train_generator = TimeseriesGenerator(
                x_train_e1,y_train_e1,
                length=win_length,
                sampling_rate=1,
                stride=1,
                batch_size=1
                )
train_generator[0]

    #preparing test sequence
test_generator = TimeseriesGenerator(
                x_test_e1,y_test_e1,
                length=win_length,
                sampling_rate=1,
                stride=1,
                batch_size=1              
                )
test_generator[0]

#%%
## BLSTM WITH ATTENTION LAYER ####
model = tf.keras.Sequential()

#Encoder / Input Layer:
sequence_input = tf.keras.layers.Input(shape=(win_length, num_features), dtype='float32')

#Bidirectional Layer
lstm_cell_size = 128
blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM
                                     (lstm_cell_size,
                                      dropout=0.3,
                                      return_sequences=True,
                                      return_state=True,
                                      recurrent_activation='relu',
                                      recurrent_initializer='glorot_uniform'), name="bi_lstm_0")(sequence_input)

blstm, forward_h, forward_c, backward_h, backward_c = tf.keras.layers.Bidirectional \
    (tf.keras.layers.LSTM
     (lstm_cell_size,
      dropout=0.2,
      return_sequences=True,
      return_state=True,
      recurrent_activation='relu',
      recurrent_initializer='glorot_uniform'))(blstm)

state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])

#Class to calculate context vector and the attention weights
class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
#Attention layer output
context_vector, attention_weights = Attention(128)(blstm, state_h)

#Decoder:
#Output layers / Decoder
output = tf.keras.layers.Dense(1, activation='relu')(context_vector)

model = tf.keras.Model(inputs=sequence_input, outputs=output)

#Compile Model: Model Evaluation - Optimization and Loss Function
model.compile(loss=tf.losses.MeanSquaredError(),
    optimizer = tf.optimizers.Adam(learning_rate = 0.1),
    metrics = 'mse'
    )
model.summary()



#%%

#Fit model
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min')
history = model.fit(train_generator, epochs=100, validation_data=test_generator, shuffle=False)

#%%
#The returned history object holds a record of the loss values and metric values during training:
history.history

#%%
#plot model history
loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)), loss_per_epoch)

#%%
#validate model
model.evaluate(test_generator,verbose=0)

#%%

predictions = model.predict(test_generator)

#%%

#prediction lentgh is less than the total number because of the rows
predictions.shape[0] #prediciton result does not have the first 20 values because of the window
predictions

#%%

predictions_2d = predictions.reshape(358,1)
predictions_2d.shape
predictions_df = pd.DataFrame(predictions_2d)

#%%
#Prediction Plots
plt.plot(predictions_df)
plt.plot(y_test_e1)

