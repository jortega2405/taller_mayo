import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from warnings import simplefilter
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

data = pd.read_csv('./DataSets/bank-full.csv')

rangos = [0, 10, 15, 25, 40, 60, 80, 100]
categories = ['1', '2', '3', '4', '5', '6', '7']
data.age = pd.cut(data.age, rangos, labels=categories)
data.job.replace(['management', 'technician', 'entrepreneur', 'blue-collar','unknown', 'retired', 'admin.', 'services', 'self-employed','unemployed', 'housemaid', 'student'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], inplace = True)
data.marital.replace(['married', 'single', 'divorced'], [0, 1, 2], inplace = True)
data.education.replace(['tertiary', 'secondary', 'unknown', 'primary'], [0, 1, 2, 3], inplace = True)
data.default.replace(['no', 'yes'], [0, 1], inplace = True)
data.housing.replace(['yes', 'no'], [1, 0], inplace = True)
data.contact.replace(['unknown', 'cellular', 'telephone'], [0, 1, 2], inplace = True)
data.loan.replace(['no', 'yes'], [0, 1], inplace = True)
data.y.replace(['no', 'yes'], [0, 1], inplace = True)
data.month.replace(['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'jan', 'feb','mar', 'apr', 'sep'], [5, 6, 7, 8, 10, 11, 12, 1, 2, 3, 4, 9], inplace = True)
data.poutcome.replace(['unknown', 'failure', 'other', 'success'], [0, 1, 2, 3], inplace = True)
data.dropna(axis=0, how='any', inplace=True)

def entrenamiento(model, x_train, x_test, y_train, y_test):
    kfold = KFold(n_splits=10)
    cvscores = []
    for train, test in kfold.split(x_train, y_train):
        model.fit(x_train[train], y_train[train])
        scores = model.score(x_train[test], y_train[test])
        cvscores.append(scores)
    y_pred = model.predict(x_test)
    accuracy_validation = np.mean(cvscores)
    accuracy_test = accuracy_score(y_pred, y_test)
    return model, accuracy_validation, accuracy_test, y_pred

def metrics(str_model, acc_validation, acc_test, y_test, y_pred):
    print('-' * 50 + '\n')
    print(str.upper(str_model))
    print('\n')
    print(f'Accuracy de validación: {acc_validation} ')
    print(f'Accuracy de test: {acc_test} ')
    print(classification_report(y_test, y_pred))
    print(f'Y real: {y_test}')
    print(f'Y predicho: {y_pred}')



def matriz_confusion(y_test, y_pred):
    matriz_confusion = confusion_matrix(y_test, y_pred)
    return matriz_confusion

x = np.array(data.drop(['y'], 1))
y = np.array(data.y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
