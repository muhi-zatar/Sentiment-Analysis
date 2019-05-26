import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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
  
def tf_idf(data, datatype):
  if datatype ==1:
    #removing stop words for better performance 
    vectorizer = TfidfVectorizer(stop_words="english",
                             ngram_range=(1, 2))
    features = vectorizer.fit_transform(reviews_train) 
  if datatype ==0:
    features = vectorizer.transform(reviews_test) 
    
  return features

def logistic_regression(training_features, labels_train, test_features, labels_test):
  for c in [0.01, 0.05, 0.25, 0.5, 1, 5]:
    #changing the parameter C to get the optimal classification
    lr = LogisticRegression(C=c)
    lr.fit(training_features, labels_train)
    print ("Accuracy of logistic regression for C=%s: %s" 
           % (c, accuracy_score(labels_test, lr.predict(test_features))))
  
def svm(training_features, labels_train, test_features, labels_test):
  for c in [1, 5, 10, 50, 100, 500]:
    #changing the parameter C to get the optimal classification
    model = LinearSVC(C=c)
    model.fit(training_features, labels_train)
    print ("Accuracy of svm for C=%s: %s" 
           % (c, accuracy_score(labels_test, model.predict(test_features))))
    
def naiive_bayes(training_features, labels_train, test_features, labels_test):
  clf = MultinomialNB()
  clf.fit(training_features, labels_train)
  print ("Accuracy of Naiive Bayes: %s" 
         % ( accuracy_score(labels_test, clf.predict(test_features))))
  
if __name__== "__main__":
  path ="imdb_master.xlsx"
  reviews_train, labels_train, reviews_test, labels_test = get_data(path)
  train = preprocess(reviews_train)
  test = preprocess(reviews_test)
  train_features = tf_idf(train, 1)
  test_features = tf_idf(test, 0)
  
  logistic_regression(training_features, labels_train, test_features, labels_test)
  svm(training_features, labels_train, test_features, labels_test)
  naiive_bayes(training_features, labels_train, test_features, labels_test)
  
  
