# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:46:10 2019

@author: anant singh
"""
#%% IMPORTS
import pandas as pd
from nltk.corpus import stopwords
from textblob import Word
import keras.utils
from sklearn.model_selection import train_test_split
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Flatten

# fix random seed for reproducibility
numpy.random.seed(7)

seed = 7
numpy.random.seed(seed)
#%% PREPROCESSING
train = pd.read_csv('poems.csv')

# making poems to lower case
train['content'] = train['content'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# removing special characters
train['content'] = train['content'].str.replace('[^\w\s]','')

# removing stopwords (a, you, my, is, etc.)
stop = stopwords.words('english')
train['content'] = train['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# remove top 10 highly frequent words
freq1 = pd.Series(' '.join(train['content']).split()).value_counts()[:10]
freq1 = list(freq1.index)
train['content'] = train['content'].apply(lambda x: " ".join(x for x in x.split() if x not in freq1))

# removing top 10 least frequent words
freq2 = pd.Series(' '.join(train['content']).split()).value_counts()[-10:]
freq2 = list(freq2.index)
train['content'] = train['content'].apply(lambda x: " ".join(x for x in x.split() if x not in freq2))

# removal of suffices, like “ing”, “ly”, “s”, etc.
train['content'] = train['content'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

vocab_size = 20000
train['content'] = [one_hot(d,vocab_size) for d in train['content']]

max_length = 500
padded_poems = pad_sequences(train['content'], maxlen=max_length, padding='post')
labels = keras.utils.to_categorical(train['type'])

x_train, x_test, y_train, y_test = train_test_split(padded_poems, labels, test_size=0.25)
# we finally have 'padded_poems' as 'X' and 'labels' as 'Y'

#%% BUILDING THE MODEL

# building a simple model
model = Sequential()
model.add(Embedding(vocab_size, 32, input_length=max_length))
model.add(Flatten())
model.add(Dense(3, activation='sigmoid'))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# summarize the model
print(model.summary())

#%% TRAINING THE MODEL

# fit the model
model.fit(x_train, y_train, epochs=100,batch_size=64, verbose=1)
# evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Accuracy: %f' % (accuracy*100))
