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

The reviews inserted by the users and fans has some undesirable characters and segements that may not be useful for training, so first of all, it is mandatory to remove these characters. Also, the preprocessing stage inc

