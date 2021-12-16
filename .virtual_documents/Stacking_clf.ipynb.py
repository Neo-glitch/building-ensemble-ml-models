import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# weak learners for ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# used as blender(or meta learner)
from sklearn.linear_model import LogisticRegression


bank_data = pd.read_csv("./datasets/bank_data_mine.csv")

bank_data.head()


X = bank_data.drop("CreditCard", axis = 1)
Y = bank_data.CreditCard


# splits dataset into 3 parts(train, holdout and test)
# 70% used for train, 20% holdout and 10% remaining for test


x_train, x_hold_out, x_test = np.split(X, [int(.7*len(X)), int(.9*len(X))])
y_train, y_hold_out, y_test = np.split(Y, [int(.7*len(X)), int(.9*len(X))])


# just to check how np.split works

# check = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# np.split(check, [int(.6*len(check)), int(.8*len(check))])


clf1 = KNeighborsClassifier(n_neighbors=10)
clf2 = RandomForestClassifier(n_estimators = 30)
clf3= GaussianNB()

# iters through and train each predictor
for clf in (clf1, clf2, clf3):
    clf.fit(x_train, y_train)


# helper fn to get predictions of each weak learner, x = x_hold_out, y = y_hold_out
def get_predictions(x, y):
    pred_result = pd.DataFrame()    # df to hold predictions of each weak learner
    
    i= 1
    for clf in (clf1, clf2, clf3):
        y_pred = clf.predict(x)
        print(clf.__class__.__name__, accuracy_score(y, y_pred))
        
        pred_result.insert(i - 1, "y_pred_" + str(i), y_pred)   # insert pred of each clf as col in pred_result df
        
        i+=1
        
    return pred_result


pred_result = get_predictions(x_hold_out, y_hold_out)

pred_result.head(10)


# data to be passed to blender for training
x_stack_train = pred_result
y_stack_train = y_hold_out


# blender or meta learner(uses predictions to train)
clf_stack = LogisticRegression(C = 1, max_iter = 200)
clf_stack.fit(x_stack_train, y_stack_train)


# blender still needs predictions of weak learners tto give final pred but using 
# weak learners prediction on test_set and not hold_out_set
pred_result_test = get_predictions(x_test, y_test)

x_stack_test = pred_result_test
y_stack_pred = clf_stack.predict(x_stack_test)


accuracy_score(y_stack_pred, y_test)
























































































































