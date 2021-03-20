import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, plot_roc_curve, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def logreg_predict_score(X_train, X_test, y_train, y_test):
    
    # make logistic regression
    logreg = LogisticRegression(fit_intercept=False, C=1e12, solver='liblinear')
    logreg.fit(X_train, y_train)

    # generate predictions
    y_train_pred = logreg.predict(X_train)
    y_test_pred = logreg.predict(X_test)

    # print scores
    print('ROC Scores')
    print('Train:', roc_auc_score(y_train, logreg.decision_function(X_train)))
    print('Test:', roc_auc_score(y_test, logreg.decision_function(X_test)))
    print('\n')
    print('Accuracy Scores')
    print('Train:', accuracy_score(y_train, y_train_pred))
    print('Test:', accuracy_score(y_test, y_test_pred))
    print('\n')
    print('F1 Scores')
    print('Train:', f1_score(y_train, y_train_pred))
    print('Test:', f1_score(y_test, y_test_pred))
    
    return logreg
    
def eval_model(X_train, X_test, y_train, y_test, model):
    
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print('ROC Scores')
    print('Train:', roc_auc_score(y_train, model.predict_proba(X_train)[:,1]))
    print('Test:', roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))
    print('\n')
    print('Accuracy Scores')
    print('Train:', accuracy_score(y_train, y_train_pred))
    print('Test:', accuracy_score(y_test, y_test_pred))
    print('\n')
    print('F1 Scores')
    print('Train:', f1_score(y_train, y_train_pred))
    print('Test:', f1_score(y_test, y_test_pred))

    plot_roc_curve(model, X_train, y_train)
    plot_roc_curve(model, X_test, y_test);
    
