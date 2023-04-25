# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pandas as pd
from wordcloud import WordCloud, STOPWORDS


df_true = pd.read_csv('True.csv')
df_fake = pd.read_csv('Fake.csv')

df_true.head()

df_fake.head()

df_true.describe()

df_fake.describe()

df_true.shape

df_fake.shape

df_true.head(10)

df_fake.head(10)

df_true.info()

df_true.isnull().sum()

df_fake.info()

df_fake.isnull().sum()

df_true['category'] = 1
df_fake['category'] = 0

df_true.head()

df_fake.head()

df = pd.concat([df_true,df_fake])
df

df.shape

df.columns

#Removing columns which are not required
df_merge = df.drop(["title", "subject","date"], axis = 1)

df_merge.isnull().sum()

df_merge.head()

#Creating a function to process the texts
import string
import re
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

df_merge["text"] = df_merge["text"].apply(wordopt)

#Defining dependent and independent variables
x = df_merge["text"]
y = df_merge["category"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

# Import the TfidfVectorizer class from the sklearn library
from sklearn.feature_extraction.text import TfidfVectorizer

# Create an instance of the TfidfVectorizer class
vectorization = TfidfVectorizer()

# Use the fit_transform method to fit the vectorization model on the training data and transform it into a matrix
# This matrix represents the term frequency–inverse document frequency (TF-IDF) values of the words in the text data
# The resulting matrix has rows for each text sample and columns for each unique word in the entire corpus
# The values in the matrix represent the weight of each word in the corresponding text sample, based on its frequency and importance in the corpus
xv_train = vectorization.fit_transform(x_train)

# Use the transform method to transform the test data into a matrix of TF-IDF values
# we use only the transform method here, not the fit_transform method, because we want to use the same vocabulary as the one learned from the training data
xv_test = vectorization.transform(x_test)

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train,y_train)

pred = LR.predict(xv_test)

LR.score(xv_test, y_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, pred))

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(xv_train, y_train)

pred_decision = model.predict(xv_test)
model.score(xv_test, y_test)

print(classification_report(y_test, pred_decision))


def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Real News"

def manual_testing(news):
    news = wordopt(news)
    news_array = np.array([news])
    news_vector = vectorization.transform(news_array)
    pred_LR = LR.predict(news_vector)
    pred_DT = model.predict(news_vector)
    print("Logistic Regression Prediction: {}".format(output_label(pred_LR[0])))
    print("Decision Tree Prediction: {}".format(output_label(pred_DT[0])))

news = str(input("Enter the news text: "))
manual_testing(news)
