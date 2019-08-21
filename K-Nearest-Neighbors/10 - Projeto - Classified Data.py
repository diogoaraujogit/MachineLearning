import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Obtendo os dados
df = pd.read_csv('KNN_Project_Data')

#Análise exploratória de dados
sns.pairplot(df,hue='TARGET CLASS',palette='coolwarm')

#Padronizando as variáveis
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

#Treinando o modelo
scaler.fit(df.drop('TARGET CLASS',axis=1))
#Versão padronizada
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))
#Convertendo em um dataframe
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])

#Divisão treino-teste
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],
                                                    test_size=0.30)

#Usando o KNN
from sklearn.neighbors import KNeighborsClassifier

#Criando uma instância com k =1
knn = KNeighborsClassifier(n_neighbors=1)

#Ajustando aos dados de treinamento
knn.fit(X_train,y_train)

#Previsões e avaliações
pred = knn.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

#Escolhendo o valor de K

error_rate = []

for i in range(1, 40):

    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

#Treinando com oo novo valor de K, K=30
knn = KNeighborsClassifier(n_neighbors=30)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=30')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))