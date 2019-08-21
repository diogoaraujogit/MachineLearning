#obtendoo os dados
import seaborn as sns
iris = sns.load_dataset('iris')

#Análise exploratória de dados
import pandas as pd
import matplotlib.pyplot as plt

#Criando um pairplot
# Setosa é a mais separável
sns.pairplot(iris,hue='species',palette='Dark2')

#KDE comprimento largura
setosa = iris[iris['species']=='setosa']
sns.kdeplot( setosa['sepal_width'], setosa['sepal_length'],
                 cmap="plasma", shade=True, shade_lowest=False)

#Divisão treino-teste
from sklearn.model_selection import train_test_split
X = iris.drop('species',axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

#Treinando o modelo
from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train,y_train)

#Avaliação do Modelo
predictions = svc_model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

#Prática do GridSearch
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)

grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))

