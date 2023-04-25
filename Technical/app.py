from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

df_true = pd.read_csv('True.csv')
df_fake = pd.read_csv('Fake.csv')

df_true['category'] = 1
df_fake['category'] = 0

df = pd.concat([df_true, df_fake])
df_merge = df.drop(["title", "subject", "date"], axis=1)


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

df_merge["text"] = df_merge["text"].apply(wordopt)

x = df_merge["text"]
y = df_merge["category"]

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x)

LR = LogisticRegression()
LR.fit(xv_train, y)

DT = DecisionTreeClassifier()
DT.fit(xv_train, y)

# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(xv_train, y)

# Multinomial Naive Bayes
nb = MultinomialNB()
nb.fit(xv_train, y)

def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Real News"


def predict(news):
    news = wordopt(news)
    news_array = np.array([news])
    news_vector = vectorization.transform(news_array)
    pred_LR = LR.predict(news_vector)
    pred_DT = DT.predict(news_vector)
    pred_RFC = rfc.predict(news_vector)
    pred_NB = nb.predict(news_vector)
    return {"Logistic Regression Prediction": output_label(pred_LR[0]),
            "Decision Tree Prediction": output_label(pred_DT[0]),
            "Random Forest Prediction": output_label(pred_RFC[0]),
            "Naive Bayes Prediction": output_label(pred_NB[0])}


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        article = request.form['article']
        result = predict(article)
        return render_template('index.html', result=result)
    else:
        result = {"Logistic Regression Prediction": "",
                  "Decision Tree Prediction": "",
                  "Random Forest Prediction": "",
                  "Naive Bayes Prediction": ""}
        return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
