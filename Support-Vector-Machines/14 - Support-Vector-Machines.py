import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Dads de câncer de mama
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
#Cnjunto de dadoos apresentado como dicionário
cancer.keys()

# pegar informações e arrays deste dicionário para configurar o dataframe e entender os recursos:
print(cancer['DESCR'])
cancer['feature_names']

# Configurando o DataFrame
df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df_feat.info()
print(cancer['target'])

df_target = pd.DataFrame(cancer['target'],columns=['Cancer'])
#df.head()

#Divisão Treino-Teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_feat, np.ravel(df_target), test_size=0.30, random_state=101)

#Treinando oo suporte vector classifier
from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)

#Previsões e Avaliações
predictions = model.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

#Modello precisa ter os parâmetros ajustados. Procurar por parâmetros usando um GridSearch

#Gridsearch
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)

grid.fit(X_train,y_train)

#Inspecionar os melhores parâmetros
#grid.best_params_
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))