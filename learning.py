#!/usr/bin/python
# -*- coding: utf-8
#Copyright 2013, AGPLv3

############################################################################
#   The script takes 5 mandatory arguments:                                #
#   learning.py [sentences] [labels] [Algorithm] [Test_size] [Vectoring]   #
#   [Algorithm] = ALL, NN, SVM, BAYES                                      #
#   [Test_size] = 0.0 .. 1.0                                               #
#   [Vectoring] = TFIDF, COUNT                                             #
############################################################################

import sys, nltk, datetime
import numpy as np
import pylab as pl
from random import shuffle
from matplotlib.colors import ListedColormap
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from sklearn import metrics
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing as prep
from sklearn.cross_validation import train_test_split

#Initial variables
now = str(datetime.datetime.now())[:-7]
categories = ["negative", "neutral", "positive"]
run = str(sys.argv[3])
size = eval(sys.argv[4])
vec = str(sys.argv[5])

""" Get, open and serve data """
#Open file and split into list
def opendatas(filename):
    with open(filename, 'r') as f:
        dict_words = f.read().splitlines()
        return dict_words
    
#Extract citing sentences
def PullSentences(datalist):
    templist = []
    for e in datalist:
        templist.append(str(eval(e)[2:-2])[2:-2])
    return templist

#Citeted sentences & labels, shuffled
def randlists(list1, list2):
    list1_shuf = []
    list2_shuf = []
    index_shuf = range(len(list1))
    shuffle(index_shuf)
    for i in index_shuf:
        list1_shuf.append(list1[i])
        list2_shuf.append(list2[i])
    return list1_shuf, list2_shuf

print "Initiating and loading data"
citedsents, citedlabels = randlists(PullSentences(opendatas(sys.argv[1])), opendatas(sys.argv[2]))

#Turning citedlabels to array
intlabels = np.array(map(int, citedlabels))    

""" Normalizing & vectorizing """
def stem(listofstr):
    st = LancasterStemmer()
    readywords = []
    for s in listofstr:
        words = nltk.word_tokenize(s)
        readywords.append(' '.join([w for w in words if not w in stopwords.words('english')]))
    return readywords

#Choose vectorizer
if vec == 'COUNT':
    print "Vectorizing with normal feature count"
    vectors = CountVectorizer() 
    X = vectors.fit_transform(stem(citedsents)).toarray()
elif vec == 'TFIDF':
    print "Vectorizing with Tf-idf"
    vectors = TfidfVectorizer(charset='utf-8')
    X = (prep.normalize(vectors.fit_transform(stem(citedsents)))).toarray()
else:
    print "Please choose how to populate the bag-of-words"
    print "use COUNT or TFIDF as fifth argument"
    raise Exception("Wrong condition!")

X_train, X_test, Y_train, Y_test = train_test_split(X, intlabels, test_size=size)

print str(len(citedsents)) + " in-situ citations in all"
print "Learning on a training set of " + str(len(X_train))
print "Test set consist of " + str(len(X_test)) + " exampels"

""" Learning """
label_names = ["negativ", "neutral", "positiv"]
def evaluation(learner):
    learner.fit(X_train, Y_train)
    outcome = learner.predict(X_test)
    return metrics.classification_report(Y_test, outcome, target_names=label_names)

def NN():
    print "=" * 52
    print "Nearest Neighbor Results:"
    for i in range(16)[1:]:
        knn_outcome = evaluation(KNeighborsClassifier(n_neighbors=i))
        print "-" * 52
        print "kNN " + "{ k = " +  str(i) + " }"
        print knn_outcome
    print "-" * 52
    print "Centroid / Rocchio"
    print evaluation(NearestCentroid())
    print "=" * 52

def SVM():
    print "=" * 52
    print "Support Vector Machine Results:"
    print "-" * 52
    for k in ['linear', 'poly', 'rbf', 'sigmoid']:
        for c in [2.0, 1.5, 1.25, 1.0, 0.75, 0.5, 0.25]:
            for g in [0.0, 0.25, 0.5, 1.0]:
                print "SVC " + "{ " + k + ": C=" + str(c) + ", gamma=" +str(g) + " }"
                print evaluation(svm.SVC(C=c,kernel=k, gamma=g))
    print "-" * 52
    print "LinearSVC"
    print evaluation(svm.LinearSVC())
    print "=" * 52

def bayes():
    print "Naive Bayes Results:"
    print "-" * 52
    print "Multinomia"
    print evaluation(MultinomialNB()) 
    print "=" * 52

""" OUTPUT """
if run == 'ALL':
    SVM()
    NN()
    bayes()
elif run == 'NN':
    NN()
elif run == 'SVM':
    SVM()
elif run == 'BAYES':
    bayes()

print "Bye!"
