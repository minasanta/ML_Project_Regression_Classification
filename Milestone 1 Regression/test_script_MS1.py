import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ast
from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder, MinMaxScaler, PolynomialFeatures
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from Preprossing import preprossing_the_test
import warnings
warnings.simplefilter('ignore')

data = pd.read_csv('movies-tas-test day 2.csv')
X = data.drop("vote_average", axis=1)
Y = data["vote_average"]

X = preprossing_the_test(X)

file = open(f"random_forest.obj", 'rb')
rf_model = pickle.load(file)
file.close()

prediction = rf_model.predict(X)

test_err = metrics.mean_squared_error(Y, prediction)
r_sqared = metrics.r2_score(Y, prediction)
print('Test subset (MSE) of random forest: ', test_err)
print('Test R^2 : ', r_sqared)

poly_features = PolynomialFeatures(degree=2)
file = open(f"ridge.obj", 'rb')
poly_with_reg = pickle.load(file)
file.close()

prediction = poly_with_reg.predict(poly_features.fit_transform(X))

test_err = metrics.mean_squared_error(Y, prediction)
r_sqared = metrics.r2_score(Y, prediction)
print('Test subset (MSE) of poly regression using ridge: ', test_err)
print('Test R^2 : ', r_sqared)