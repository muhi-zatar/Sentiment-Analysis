# Sentiment Analysis

Sentiment Analysis, which is also known as opinion mining is the process of labeling or classifying a given review if it is positive(good) or negative(bad). Sentiment analysis is a subject of great interest, since it has many practical applications. Since publicly and privately available information over Internet is constantly growing, a large number of texts expressing opinions are available in review sites, forums, blogs, and social media.

Sentiment analysis can be thought of as a binary classification problem; simply a given review is required to be classified as a positive review or a bad review. In this repositary, different techniques and approaches are explained, implemented and compared in terms of accuracy of classification and speed of testing and training. 

This repositary is written in the following flow:

1- Dataset

2- Data Preprocessing

3- Machine Learning approaches.

4- Deep Learning approaches.

5- Results comparisons.

# Prequisits

- Python 2.7 or 3.6.
- tensorflow
- Sckilearn

# Dataset

People tend to express their opinions and reviews regurarly about the movies and series they have watched, and imdb is a popular forum for this purpose. In this exercise, the imdb dataset will be used for sentiment analysis.

The dataset is divided into training set, which is 25k sentences, and a testing set which as 25k sentences. Also, the dataset includes other 50k unlabeled reviews, which can be used for unsupervised learning. 

In this repository, we will first evaluate different techniques by training the learning algorithms using the labeled training data, and then add the unsupervised data to the training set and compare the results before and after.

# Data Preprocessing

The reviews inserted by the users and fans has some undesirable characters and segements that may not be useful for training, so first of all, it is mandatory to remove these characters. Also, the preprocessing stage is done easily using regular expressions (https://www.w3schools.com/python/python_regex.asp).

# Tf-idf

One way of representing words by numbers is tf-idf, and gives better results the one hot encoding and sometimes on some datasets, it outperforms more sophisticated word representations such as word2vec or fasttext.
Tf-idf stands for term frequency, inverse document frequency. From our intuition, we think that the words which appear more often should have a greater weight in textual data analysis, but that’s not always the case. Words such as “the”, “will”, and “you” — called stopwords — appear the most in a corpus of text, but are of very little significance. Instead, the words which are rare are the ones that actually help in distinguishing between the data, and carry more weight. Thats how basically tf-idf works. 

# Machine learning Techniques
In this section, multiple machine learning techniques were used for the sake of sentiment analysis. Logistic regression, Support Vector Machines (SVM) and Naiive Bayes were implemented and compared. 

**Logistic Regression**




