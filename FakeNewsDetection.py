import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


dataframe =pd.read_csv("news.csv")

print(dataframe.shape)
print(dataframe.head(5))
labels =dataframe.label
print(labels.shape)
X_train,X_test,y_train,y_test =train_test_split(dataframe["text"],labels,test_size=0.2,random_state=7)
print(X_train.shape)
print(y_train.shape)

tfidf_vectorizer = TfidfVectorizer(stop_words='english',max_df=0.7)
tfidf_train=tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

pac=PassiveAggressiveClassifier(max_iter=50)

print(tfidf_train.shape)

y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')
confusion_matrix(y_test,y_pred,labels=['FAKE','REAL'])
