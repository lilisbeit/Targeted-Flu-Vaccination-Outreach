import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, plot_roc_curve, accuracy_score, f1_score, plot_confusion_matrix
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
    print('\n')
    print('Cross-Validation Scores:')
    print('Train:')
    print(cross_val_score(logreg, X_train, y_train, scoring='roc_auc'))
    print('Mean:', round(cross_val_score(logreg, X_train, y_train, scoring='roc_auc').mean(), 3))
    print('Test:')
    print(cross_val_score(logreg, X_test, y_test, scoring='roc_auc'))
    print('Mean:', round(cross_val_score(logreg, X_test, y_test, scoring='roc_auc').mean(),3))
    
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
    plot_roc_curve(model, X_test, y_test)
    plot_confusion_matrix(model, X_train, y_train)
    plot_confusion_matrix(model, X_test, y_test)

def roc_auc_cross_val(X_train, X_test, y_train, y_test, model):

    print('Train ROC-AUC Cross Validation:', cross_val_score(model, X_train, y_train, scoring='roc_auc'))
    print('Train Mean:', round(cross_val_score(model, X_train, y_train, scoring='roc_auc').mean(), 3))
    print('\n')
    print('Test ROC-AUC Cross Validation:', cross_val_score(model, X_test, y_test, scoring='roc_auc'))
    print('Test Mean:', round(cross_val_score(model, X_test, y_test, scoring='roc_auc').mean(),3))
    

def order_features(weights, X_train):
    
    coef_dict = {}

    for n, c in enumerate(X_train.columns):
        coef_dict[c]=round(weights[n],4)

    sorted_coef_dict = {k: v for k, v in sorted(coef_dict.items(), key=lambda item: item[1], reverse=True)}
    df = pd.DataFrame.from_dict(sorted_coef_dict, orient='index', columns=['weight'])
    df['abs_weight']=np.abs(df['weight'])
    weights_df = df.sort_values(by = 'abs_weight', ascending=False)
    
    return weights_df


    
