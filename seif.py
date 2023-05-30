import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

import mlflow
import mlflow.sklearn
 
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

df = pd.read_csv('pointure.data')

import numpy as np
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
input_classes = ['masculin','féminin']
label_encoder.fit(input_classes)

# transformer un ensemble de classes
encoded_labels = label_encoder.transform(df['Genre'])

df['Genre'] = encoded_labels

X = df.iloc[:, lambda df: [1, 2, 3]]
y = df.iloc[:, 0]

from sklearn.model_selection import train_test_split

#decomposer les donnees predicteurs en training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_naive_bayes1 = gnb.predict(X_train)

#EVALUATION SUR LE TRAIN
print("Number of mislabeled points out of a total 0%d points : 0%d" % (X_train.shape[0],(y_train != y_naive_bayes1).sum()))
accuracyTrain = metrics.accuracy_score(y_train, y_naive_bayes1)
print("Accuracy du modele Naive Bayes predit: " + str(accuracyTrain))


recall_scoreTrain = metrics.recall_score(y_train, y_naive_bayes1)
print("recall score du modele Naive Bayes predit: " + str(recall_scoreTrain))

f1_scoreTrain = metrics.f1_score(y_train, y_naive_bayes1)
print("F1 score du modele Naive Bayes predit: " + str(f1_scoreTrain))

#EVALUATION SUR LE TEST
y_naive_bayes2 = gnb.predict(X_test)
print("Number of mislabeled points out of a total 0%d points : 0%d" % (X_test.shape[0],(y_test != y_naive_bayes2).sum()))
accuracyTest = metrics.accuracy_score(y_test, y_naive_bayes2)
print("Accuracy du modele Naive Bayes predit: " + str(accuracyTest))

recall_scoreTest = metrics.recall_score(y_test, y_naive_bayes2)
print("recall score du modele Naive Bayes predit: " + str(recall_scoreTest))

f1_scoreTest = metrics.f1_score(y_test, y_naive_bayes2)
print("F1 score du modele Naive Bayes predit: " + str(f1_scoreTest))