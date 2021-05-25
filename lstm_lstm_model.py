import numpy as np
import pickle
import pandas as pd

import tensorflow.keras as keras

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Masking, LSTM, TimeDistributed, \
                         Bidirectional, Flatten, \
                         Embedding, Dropout, Flatten, BatchNormalization, \
                         RNN, concatenate, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import sequence
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import random as rn
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop

rn.seed(123)
np.random.seed(99)
tf.compat.v1.set_random_seed(1234)

# load audio features
feat = np.load('/Users/Evan/Desktop/Click/feat_hsf.npy')
# load VAD scores
vad = np.load('/Users/Evan/Desktop/Click/y_egemaps.npy')
# remove outlier, < 1, > 5
vad = np.where(vad==5.5, 5.0, vad)
vad = np.where(vad==0.5, 1.0, vad)
# standardization
scaled_feature = False

# text feature
x_train_text = np.load('/Users/Evan/Desktop/Click/x_train_text.npy')
g_word_embedding_matrix = np.load('/Users/Evan/Desktop/Click/g_word_embedding_matrix.npy')

# other parameters
MAX_SEQUENCE_LENGTH = 500
EMBEDDING_DIM = 300
nb_words = 3438

if scaled_feature == True:
    scaler = StandardScaler()
    scaler = scaler.fit(feat.reshape(feat.shape[0]*feat.shape[1], feat.shape[2]))
    scaled_feat = scaler.transform(feat.reshape(feat.shape[0]*feat.shape[1], feat.shape[2]))
    scaled_feat = scaled_feat.reshape(feat.shape[0], feat.shape[1], feat.shape[2])
    feat = scaled_feat
else:
    feat = feat

scaled_vad = True

# standardization
if scaled_vad:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(vad) #.reshape(vad.shape[0]*vad.shape[1], vad.shape[2]))
    scaled_vad = scaler.transform(vad) #.reshape(vad.shape[0]*vad.shape[1], vad.shape[2]))
    vad = scaled_vad 
else:
    vad = vad

def CCC(y_true, y_pred):
    '''Lin's Concordance correlation coefficient: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    
    The concordance correlation coefficient is the correlation between two variables that fall on the 45 degree line through the origin.
    
    It is a product of
    - precision (Pearson correlation coefficient) and
    - accuracy (closeness to 45 degree line)

    Interpretation:
    - `rho_c =  1` : perfect agreement
    - `rho_c =  0` : no agreement
    - `rho_c = -1` : perfect disagreement 
    
    Args: 
    - y_true: ground truth
    - y_pred: predicted values
    
    Returns:
    - concordance correlation coefficient (float)
    '''
    
    import keras.backend as K 
    # covariance between y_true and y_pred
    N = K.int_shape(y_pred)[-1]
    s_xy = K.mean(K.sum((y_true - K.mean(y_true)) * (y_pred - K.mean(y_pred))))
    # means
    x_m = K.mean(y_true)
    y_m = K.mean(y_pred)
    # variances
    s_x_sq = K.var(y_true)
    s_y_sq = K.var(y_pred)
    
    # condordance correlation coefficient
    ccc = (2.0*s_xy) / (s_x_sq + s_y_sq + (x_m-y_m)**2)
    
    return ccc


def model(alpha, beta, gamma):
    # speech network
    input_speech = Input(shape=(feat.shape[1], feat.shape[2]), name='speech_input')
    net_speech = BatchNormalization()(input_speech)
    net_speech = LSTM(256, return_sequences=True)(net_speech)
    net_speech = LSTM(256, return_sequences=True)(net_speech)
    net_speech = LSTM(256, return_sequences=True)(net_speech)
    net_speech = Flatten()(net_speech)
    model_speech = Dropout(0.3)(net_speech)
    #text network
    input_text = Input(shape=(MAX_SEQUENCE_LENGTH, ))
    net_text = Embedding(nb_words,
                    EMBEDDING_DIM,
                    weights = [g_word_embedding_matrix],
                    trainable = True)(input_text)
    net_text = LSTM(256, return_sequences=True)(net_text)
    net_text = LSTM(256, return_sequences=True)(net_text)
    net_text = LSTM(256, return_sequences=False)(net_text)
    net_text = Dense(64)(net_text)
    model_text = Dropout(0.3)(net_text)
    # combined model
    model_combined = concatenate([model_speech, model_text])
    model_combined = Dense(64, activation='relu')(model_combined)
    model_combined = Dense(32, activation='relu')(model_combined)
    model_combined = Dropout(0.4)(model_combined)
    target_names = ('v', 'a', 'd')
    model_combined = [Dense(1, name=name)(model_combined) for name in target_names]

    model = Model([input_speech, input_text], model_combined) 
    #model.compile(loss=ccc_loss,
                  #loss_weights={'v': alpha, 'a': beta, 'd': gamma},
                  #optimizer='RMSprop', metrics=[ccc])
    #model.compile(loss_weights={'v': alpha, 'a': beta, 'd': gamma},
                  #optimizer='RMSprop', metrics=[CCC])
    model.compile(loss='mse', loss_weights={'v': alpha, 'a': beta, 'd': gamma},
                  optimizer='RMSprop', metrics=[CCC, 'accuracy'])
    return model
    
model = model(1, 1, 1)
model.summary()

earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=10,
                          restore_best_weights=True)
hist = model.fit([feat[:7869], x_train_text[:7869]], 
                  vad[:7869], batch_size=8, validation_split=0.2, #best:8
                  epochs=30, verbose=1, shuffle=True,
                  callbacks=[earlystop])
metrik = model.evaluate([feat[7869:], x_train_text[7869:]], vad[7869:].T.tolist())
print("CCC: ", metrik[-3:]) # np.mean(metrik[-3:]))
print("CCC_mean: ", np.mean(metrik[-3:]))
