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
simplefilter(action='ignore', category=FutureWarning)

data = pd.read_csv('./DataSets/diabetes.csv')

rangos = [0, 10, 15, 25, 40, 60, 80, 100]
categories = ['1', '2', '3', '4', '5', '6', '7']
data.Age = pd.cut(data.Age, rangos, labels=categories)

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

x = np.array(data.drop(['Outcome'], 1))
y = np.array(data.Outcome)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


model1 = DecisionTreeClassifier()
model1, accuracy_test1, accuracy1, y_predict1 = entrenamiento(model1, x_train, x_test, y_train, y_test)
metrics('Arbol de decisión', accuracy_test1, accuracy1, y_test, y_predict1)
matriz_confusion1 = matriz_confusion(y_test, y_predict1)
matriz_confusion1 = pd.DataFrame(matriz_confusion1)
sns.heatmap(matriz_confusion1)

model2 = KNeighborsClassifier()
model2, accuracy_test2, accuracy2, y_predict2 = entrenamiento(model2, x_train, x_test, y_train, y_test)
metrics('Vecinos mas cercanos', accuracy_test2, accuracy2, y_test, y_predict2)
matriz_confusion2 = matriz_confusion(y_test, y_predict2)
matriz_confusion2 = pd.DataFrame(matriz_confusion2)
sns.heatmap(matriz_confusion2)

model3 = LogisticRegression(solver='lbfgs', max_iter=10000)
model3, accuracy_test3, accuracy3, y_predict3 = entrenamiento(model3, x_train, x_test, y_train, y_test)
metrics('Regresion Logistica', accuracy_test3, accuracy3, y_test, y_predict3)
matriz_confusion3 = matriz_confusion(y_test, y_predict3)
matriz_confusion3 = pd.DataFrame(matriz_confusion3)
sns.heatmap(matriz_confusion3)

model4 = GaussianNB()
model4, accuracy_test4, accuracy4, y_predict4 = entrenamiento(model4, x_train, x_test, y_train, y_test)
metrics('Gaussian NB', accuracy_test4, accuracy4, y_test, y_predict4)
matriz_confusion4 = matriz_confusion(y_test, y_predict4)
matriz_confusion4 = pd.DataFrame(matriz_confusion4)
sns.heatmap(matriz_confusion4)

model5 = RandomForestClassifier()
model5, accuracy_test5, accuracy5, y_predict5 = entrenamiento(model5, x_train, x_test, y_train, y_test)
metrics('Random Forest Clasifier', accuracy_test5, accuracy5, y_test, y_predict5)
matriz_confusion5 = matriz_confusion(y_test, y_predict5)
matriz_confusion5 = pd.DataFrame(matriz_confusion5)
sns.heatmap(matriz_confusion5)