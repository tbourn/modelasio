# tested by flake8 (https://pypi.org/project/pytest-flake8/)

# Importing the libraries
from __future__ import division
import os
import lasio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# Plotting function
def plot(model, prediction, color):
    plt.close()
    plt.figure(figsize=plt.figaspect(0.5))
    plt.title(model, fontsize=18)
    plt.xlabel('Predicted Values', fontsize=14)
    plt.ylabel('Actual Values', fontsize=14)
    plt.scatter(prediction, y_test, color='%s' % color)
    plt.savefig('figures/%s' % model)
    plt.clf()

'''
# Cross Validation
def cross_validation(model, X_train, y_train):
    scores = cross_val_score(model, X_train, y_train, cv=50)
    print(2 * '\n', '%s accuracy: %0.2f (+/- %0.2f)' % (
            model, scores.mean(), scores.std() * 2))
'''

# Creating figures folder
fig_directory = './figures'
if not os.path.exists(fig_directory):
    os.makedirs(fig_directory)

# Importing the dataset and removing rows containing -999.2500; i.e NaNs.
las = lasio.read('./WLC_COMPOSITE_1.las')
df = las.df()
X = pd.DataFrame(data=df.iloc[:, [6, 0, 7, 9, 1]])
X1 = X.dropna().values
X = X1[:, 0:4]
y = X1[:, 4]

# Scaling features using statistics that are robust to outliers.
sc_X = RobustScaler()
X = sc_X.fit_transform(X)
y = y.reshape(y.shape[0])

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

# Fitting Support Vector Regression to the dataset
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)

# Fitting Random Forests Regression to the dataset
rf = RandomForestRegressor(n_estimators=300, random_state=0)
rf.fit(X_train, y_train)
'''
# Validating data
cross_validation(svr, X_train, y_train)
cross_validation(rf, X_train, y_train)
'''
# Predicting a new result
y_pred_svr = svr.predict(X_test)
y_pred_rf = rf.predict(X_test)

# Computing the Coefficient of Determination
r2_svr = r2_score(y_test, y_pred_svr)
r2_rf = r2_score(y_test, y_pred_rf)
print(2 * '\n', 'R-squared(Support Vector Regression) = ', r2_svr)
print('\n', 'R-squared(Random Forests Regression) = ', r2_rf)

# Finding the best values for parameters of the given model
print(2 * '\n', 'Support Vector Regression best fit values',
      '\n', 'weights = ', np.matmul(svr.dual_coef_, svr.support_vectors_))
print('intercept = ', svr.intercept_, 2 * '\n')

# Visualising the results
plot('Support Vector Regression', y_pred_svr, 'red')
plot('Random Forests Regression', y_pred_rf, 'blue')
