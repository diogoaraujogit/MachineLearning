import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Informações do cliente, como Email, Endereço, sua cor de  Avatar e dados de uso
customers = pd.read_csv("Ecommerce Customers")

# Visualizando
customers.head()
customers.describe()
#customers.info()

#Análise de dados exploratória

# Comparando Time on Website e Volume Anual
sns.set_palette("GnBu_d")
sns.set_style('whitegrid')
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)
# O mesmo, porém com a coluna 'Time on App'
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)
#Usando o apirplot para encontrar relações
sns.pairplot(customers)

#quantia anual gasta (Yearly Amount Spent) vs. tempo de associação (Length of Membership).
plt.close()
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)
plt.show()

# Treinando e testando dados
#Dividindo os dados
X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Construção do modelo
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

#Treinando
lm.fit(X_train,y_train)
print('Coefficients: \n', lm.coef_)

#Avaliando o desempenho de predição
predictions = lm.predict( X_test)
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

#Avaliando o modelo
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

#Resíduos
sns.distplot((y_test-predictions),bins=50);

#Conclusão
#Criando um novo quadro de dados
coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
