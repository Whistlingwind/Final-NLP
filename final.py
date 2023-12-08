#Import libraries
from nltk.util import pr
import pandas as pd
import numpy as np
# import CountVectorizer, which enables transform of text into vectors on the basis of frequency counts
from sklearn.feature_extraction.text import CountVectorizer
# import training functions
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#Extract data, using a generic twitter postings, which contain many types of speech and hate speech
data = pd.read_csv('C:\\Users\\Whistlingwind\\Desktop\\DataScience\\Phase2\\Final_Assignment\\twitter.csv')

#Add labels to the level of offense of that tweet
data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive"})
#print(data.head())

#Recreate dataframe with only tweets and label columns
data = data[["tweet", "labels"]]
#print(data.head())

import re
import nltk
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
#Collect and collate stopwords, using the english library
stopword=set(stopwords.words('english'))

#Remove characters that are not related to the message and are just either noise, or tweet structure or bad punctuation etc.
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["tweet"] = data["tweet"].apply(clean)
#print(data.head())

#Prepare data for training
x = np.array(data["tweet"])
y = np.array(data["labels"])

#Fit the data via count vectorizer, and create train and test datasets
cv = CountVectorizer()
X = cv.fit_transform(x) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Use decision tree algorithm to sort and train
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

#Create streamlit interface to allow for manual testing against trained data.
def hate_speech_detection():
    import streamlit as st
    st.title("Hate Speech Detection")
    user = st.text_area("Enter any Tweet: ")
    if len(user) < 1:
        st.write("  ")
    else:
        sample = user
        data = cv.transform([sample]).toarray()
        a = clf.predict(data)
        st.title(a)
hate_speech_detection()