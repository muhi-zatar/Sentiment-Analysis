import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
import re

def get_data(path):
  data = pd.read_excel(path)
  test_data = data[:25000]
  train_data = data[25000:50000]
  shuffled_train = train_data.sample(frac=1)
  reviews_train = shuffled_train['review'].values
  #mapping 0 to negative and 1 to positive
  labels_train = (shuffled_train['label'].map({'neg':0, 'pos':1}).astype(int)).values
  reviews_test = test_data['review'].values
  labels_test = (test_data['label'].map({'neg':0, 'pos':1}).astype(int)).values
  
  return  reviews_train, labels_train, reviews_test, labels_test

def preprocess_data(reviews):
  #preprocessing

  for i in range(len(reviews)):
  # remove HTML tags
    reviews[i] = re.sub(r'<.*?>', '', reviews[i])
    
  # remove the characters 
    reviews[i] = re.sub(r"\\", "", reviews[i])    
    reviews[i] = re.sub(r"\'", "", reviews[i])    
    reviews[i] = re.sub(r"\"", "", reviews[i])    
    reviews[i] = re.sub(r"\,", "", reviews[i])    
    reviews[i] = re.sub(r"\.", "", reviews[i])
  # convert text to lowercase
    reviews[i] = reviews[i].strip().lower()
    
  return reviews

def pad(data):
  max_fatures = 2000
  tokenizer = Tokenizer(num_words=max_fatures, split=' ')
  tokenizer.fit_on_texts(data)
  X = tokenizer.texts_to_sequences(data)
  X = pad_sequences(X)
  return X

def build_model(X_train, Y_train, X_test, Y_test):
  embed_dim = 128
  lstm_out = 196
  model = Sequential()
  model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
  model.add(SpatialDropout1D(0.4))
  model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
  model.add(Dense(2,activation='softmax'))
  model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
  print(model.summary())
  batch_size = 128
  model.fit(X_train, Y_train, epochs = 15, batch_size=batch_size, verbose = 1)
  Y_pred = model.predict_classes(X_test,batch_size = batch_size)
  print ("Accuracy of LSTM: %s" 
         % ( accuracy_score(labels_test, Y_pred)))
  return model

if __name__== "__main__":
  path ="imdb_master.xlsx"
  reviews_train, labels_train, reviews_test, labels_test = get_data(path)
  train = preprocess(reviews_train)
  test = preprocess(reviews_test)
  train = pad(train)
  test = pad(test)
  model = build_model(train,labels_train, test,labels_test)
  
  
  
