import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Informações sobre casas em regiões dos USA
USAhousing = pd.read_csv('USA_Housing.csv')
USAhousing.head()
#USAhousing.info()
#print(USAhousing.columns)

# Alguns gráficos para poder visualizar como são as correlações plt.show()
sns.pairplot(USAhousing)
sns.distplot(USAhousing['Price'])
sns.heatmap(USAhousing.corr())

# Dividindo nossos dados em uma matriz X que contém os recursos para treinar, e uma matriz y com a variável alvo
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']

# Dividindo os dados em um conjunto de treinamento e um conjunto de testes
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
# Relação da quantidade de dados que vão servir pra treino e as que vão servir de teste é o test_size

#Criando e treinando o modelo
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)

#Avaliação do modelo
# Printando a intercepção e os coeficientes
#print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
#print(coeff_df)

# testando as previsões do modelo. plt.show() e plt.close()
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=50);

from sklearn import metrics
#Média do valor absoluto dos erros
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
#Média dos erros quadrados
print('MSE:', metrics.mean_squared_error(y_test, predictions))
#Raiz do erro quadrático médio
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
