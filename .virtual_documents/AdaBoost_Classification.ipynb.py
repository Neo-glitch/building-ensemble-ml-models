import pandas as pd

bank_data = pd.read_csv("./datasets/bank_data_mine.csv")

bank_data.head()


X = bank_data.drop("CreditCard", axis = 1)

Y = bank_data.CreditCard


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1),  # trees with depth = 1 are called stumps
                            n_estimators = 100,
                             # SAMME.R is better, and uses class prob instead of predictions from estimators to decide output
                            algorithm="SAMME",  
                            learning_rate=1.0)

ada_clf.fit(x_train, y_train)


ada_clf.estimator_weights_


y_pred = ada_clf.predict(x_test)


from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)












































































































