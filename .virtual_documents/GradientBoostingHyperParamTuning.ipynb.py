import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor


concrete = pd.read_csv("./datasets/concrete_data.csv")

concrete.head()


X = concrete.drop("csMPa", axis = 1)
Y = concrete.csMPa


X = X.drop(["flyash", "coarseaggregate", "fineaggregate"], axis = 1)  # done to reduce dim, and help increase speed


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)


# baseline model, i.e new model acc mustn't be lower than this
baseline = GradientBoostingRegressor(max_depth=3, n_estimators=50)
baseline.fit(x_train, y_train)


from sklearn.metrics import r2_score

y_pred = baseline.predict(x_test)

r2_score(y_test, y_pred)


# get importance of feature to the model
important_features = pd.Series(baseline.feature_importances_, index = X.columns).sort_values(ascending = False)

important_features


# gridSearch
gbr = GradientBoostingRegressor(max_depth=3)

parameters = {"n_estimators": [1, 5, 10, 50, 100, 200, 300, 400, 500]}

gridsearch_reg = GridSearchCV(gbr, param_grid=parameters, cv = 3, n_jobs=-1)

gridsearch_reg.fit(x_train, y_train)


gridsearch_reg.best_params_


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)


gbr_best = GradientBoostingRegressor(max_depth=3, n_estimators=gridsearch_reg.best_params_["n_estimators"])
gbr_best.fit(x_train, y_train)


y_pred = gbr_best.predict(x_test)

r2_score(y_test, y_pred)


"""
warmstart = True, means reuse solutions of previous call to predictor to fit and add
more estimators of ensemble
"""
gbr = GradientBoostingRegressor(max_depth = 3, warm_start = True)


# early stopping code
min_val_error = float("inf")
error_increasing = 0

for n_estimators in range(1, 1000):
    gbr.n_estimators = n_estimators
    gbr.fit(x_train, y_train)
    
    y_pred = gbr.predict(x_test)
    val_error = mean_squared_error(y_test, y_pred)
    
    print("No. of estimators: ", gbr.n_estimators)
    print("Error: ", val_error)
    
    if val_error < min_val_error:  # error reducing
        min_val_error = val_error  # updates min error value
        error_increasing = 0
    else:
        error_increasing += 1
        if error_increasing == 10:   # error keeps increasing for 6 times i.e overfitting
            break
        


n_estimators # best num of estimators from early stopping


# reshuffle train and test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)


gbr_best = GradientBoostingRegressor(max_depth = 3, n_estimators = n_estimators)

gbr_best.fit(x_train, y_train)


y_pred = gbr_best.predict(x_test)

r2_score(y_test, y_pred)




























































