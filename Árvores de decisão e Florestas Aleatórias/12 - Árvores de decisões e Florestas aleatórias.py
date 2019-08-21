import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dados kyphosis
df = pd.read_csv('kyphosis.csv')

# Análise exploratória de dados
sns.pairplot(df,hue='Kyphosis',palette='Set1')

# Divisão Treino-teste
from sklearn.model_selection import train_test_split

X = df.drop('Kyphosis',axis=1)
y = df['Kyphosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Árvore de decisão
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

#Previsão e Avaliação
predictions = dtree.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

# Florestas Aleatórias
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))

print(classification_report(y_test,rfc_pred))