import pandas as pd
import numpy as np
import random
import time
import csv
import pickle
import torch
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

import matplotlib.pyplot as plt
import re
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


urls_data = pd.read_csv("data.csv")


type(urls_data)
urls_data.head()

def makeTokens(f):
    tkns_BySlash = str(f.encode('utf-8')).split('/')
    total_Tokens = []
    for i in tkns_BySlash :
        tokens = str(i).split('-')
        tkns_ByDot = []
        for j in range(0, len(tokens)):
            temp_Tokens = str(tokens[j]).split('.')
            tkns_ByDot = tkns_ByDot + temp_Tokens
        total_Tokens = total_Tokens + tokens + tkns_ByDot
    total_Tokens = list(set(total_Tokens))
    if 'com' in total_Tokens:
        total_Tokens.remove('com')
    return total_Tokens

start = time.time() # save starting point1
y = urls_data["label"]
url_list = urls_data["url"]

#vectorizer = TfidfVectorizer(tokenizer=makeTokens, analyzer = 'char')
vectorizer = TfidfVectorizer(tokenizer=makeTokens)
X = vectorizer.fit_transform(url_list)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


grad = GradientBoostingClassifier(learning_rate = 0.5, max_depth = 8
                                  , n_estimators = 100, min_samples_leaf = 14, min_samples_split = 10).fit(X_train, y_train)
joblib.dump(grad, 'gradi.pkl')
