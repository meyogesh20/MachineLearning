import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import RidgeCV
#from sklearn.metrics import make_scorer


df = pd.read_csv("D:\Data Science 2024\Study\Advertising.csv")

X = df.drop('Sales',axis=1)
y = df['Sales']

poly_converter = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly_converter.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(poly_features,y,test_size=0.3,random_state=101)

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

ridge_model = Ridge(alpha=10)

ridge_model.fit(X_train,y_train) 

ridge_predictions = ridge_model.predict(X_test)

MAE = mean_absolute_error(y_test, ridge_predictions)

RMSE = np.sqrt(mean_squared_error(ridge_predictions,y_test))

ridgeCV_model = RidgeCV(alphas=(0.1, 1.0, 10.0),scoring='neg_mean_absolute_error') #Used default values of alpha/lambda

ridgeCV_model.fit(X_train,y_train)

optimal_Alpha = ridgeCV_model.alpha_

test_predictions = ridgeCV_model.predict(X_test)

mae = mean_absolute_error(y_test, test_predictions)
rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

coeff = ridgeCV_model.coef_
