import pandas as pd
import matplotlib.pyplot as plt

concrete = pd.read_csv("./datasets/concrete_data.csv")

concrete.head()


X = concrete.drop("csMPa", axis = 1)
Y = concrete.csMPa


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)


from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor


ada_reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 4),
                           n_estimators=100,
                           learning_rate = 1.0)  # note low lr, will req more estimators from train data

ada_reg.fit(x_train, y_train)


# gets predictors of each estimator and wieghs them using prdictor weights
ada_reg.estimator_weights_


y_pred = ada_reg.predict(x_test)


from sklearn.metrics import r2_score

r2_score(y_test, y_pred)

































































































































