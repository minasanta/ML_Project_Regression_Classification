import pandas as pd
import numpy as np
import ast
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import f_regression, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from Preprossing import preprossing_the_test
import warnings
warnings.simplefilter('ignore')

data = pd.read_csv('movies-tas-test day 2.csv')
data["Rate"] = data["Rate"].replace(['Low', 'Intermediate', 'High'], [1, 2, 3])
X = data.drop("Rate", axis=1)
Y = data["Rate"]

X = preprossing_the_test(X)

file = open("RandomForestClassifier.obj", "rb")
clf = pickle.load(file)
file.close()

print('Test Accuracy using RandomForestClassifier : ' +
      str(clf.score(X, Y)*100))

file = open("SVM.obj", "rb")
svm = pickle.load(file)
file.close()

print('Test Accuracy using SVM with RBF Kernel: ' +
      str(svm.score(X, Y)*100))

file = open("OneVsOneClassifier.obj", "rb")
lr_ovo = pickle.load(file)
file.close()

print('Test Accuracy using OneVsOne Logistic Regression: ' +
      str(lr_ovo.score(X, Y)*100))