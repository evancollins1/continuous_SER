import os
import sys

import wave
import copy
import math
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import pickle
import codecs

code_path = os.path.dirname(os.path.realpath(os.getcwd()))
data_path = '/Users/Evan/Desktop/IEMOCAP_full_release'
sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
framerate = 16000

with open('/Users/Evan/Desktop/IEMOCAP_full_release/data_collected_10039.pickle', 'rb') as handle:
    data2 = pickle.load(handle)
    
text = []

for ses_mod in data2:
    text.append(ses_mod['transcription'])

MAX_SEQUENCE_LENGTH = 500

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)

token_tr_X = tokenizer.texts_to_sequences(text)
x_train_text = []

x_train_text = sequence.pad_sequences(token_tr_X, maxlen=MAX_SEQUENCE_LENGTH)

np.save('/Users/Evan/Desktop/x_train_text', x_train_text)

EMBEDDING_DIM = 300

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

file_loc = '/Users/Evan/Desktop/glove.840B.300d.txt'

print (file_loc)

gembeddings_index = {}
with codecs.open(file_loc, encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        gembedding = np.asarray(values[1:], dtype='float32')
        gembeddings_index[word] = gembedding
#
f.close()
print('G Word embeddings:', len(gembeddings_index))

nb_words = len(word_index) +1
g_word_embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    gembedding_vector = gembeddings_index.get(word)
    if gembedding_vector is not None:
        g_word_embedding_matrix[i] = gembedding_vector
        
print('G Null word embeddings: %d' % np.sum(np.sum(g_word_embedding_matrix, axis=1) == 0))

np.save('/Users/Evan/Desktop/g_word_embedding_matrix.npy', g_word_embedding_matrix)

