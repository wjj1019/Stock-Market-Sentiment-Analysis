import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, SpatialDropout1D
from tensorflow.keras import regularizers
import keras
from tensorflow import keras
from gensim.models import KeyedVectors
from gensim import models
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Import data 
df = pd.read_csv('processed.csv')
df.drop(columns = ['Unnamed: 0'], inplace = True)

#Split into Dependent and Independent
X = df['Text']
y = pd.get_dummies(df['Sentiment'])

#Import Pretraiend Word2Vec
pretrained = KeyedVectors.load_word2vec_format('https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz',binary=True,limit=100000)


#Tokenizing and Padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)

sequence = tokenizer.texts_to_sequences(X)
padded = pad_sequences(sequence, maxlen=23)

word_index = tokenizer.word_index
vocab_size = len(word_index) +1


#Function to create embedding matrix
def embedding_matrix (dimension, vocab_size, word2vec, word_index):
    
    embedding_dim = dimension # Embedding Dimensions
    embedding_matrix = np.zeros((vocab_size, embedding_dim)) #Initializing Embedding Matrix (Weight Matrix for Embedding Layer)
    embedding_vector = 0 #Initialization of vector for each word 

    #Iterate through word_index obtained from Keras
    for word, index in word_index.items():
        try:
            embedding_vector = word2vec[word] #Extract 300dim vector from pretrained word Embedding
        except:
            pass #if the word within the word_index not present, leave it as 0 vector
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector #Place embedding vector into embedding matrix
    print('Embedding Matrix Shape:', embedding_matrix.shape)
    
    return embedding_matrix 

#Computing Embedding Matrix
embedding_matrix = embedding_matrix(300, vocab_size, pretrained, word_index)

#LSTM Model
rnn = Sequential()
rnn.add(Embedding(input_dim = vocab_size, output_dim = 300, input_length = 23, 
                embeddings_initializer = keras.initializers.Constant(embedding_matrix),
                trainable = False))

rnn.add(SpatialDropout1D(0.5))
rnn.add(LSTM(258, dropout=0.2, recurrent_dropout=0.2))
rnn.add(Dense(32, activation = 'relu',kernel_regularizer=regularizers.l1_l2(l1 = 0.01, l2 = 0.01)))
rnn.add(Dropout(0.5))
rnn.add(Dense(2, activation='softmax'))
rnn.compile(loss = 'categorical_crossentropy', optimizer='adam' ,metrics = ['accuracy'])

#Train Model
rnn.fit(padded, y, epochs = 20, batch_size = 36, verbose = 1)

rnn.save('model.h5')
