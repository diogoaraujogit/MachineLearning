import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('titanic_train.csv')

#Análise de dados exploratórios
#Dados ausentes -  criar um mapa de calor simples para ver onde estamos perdendo dados
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#Contagem de sobreviventes
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')

#Distribuiçao por idade
train['Age'].hist(bins=30,color='darkred',alpha=0.7)

#Limpando os dados
# Preenchendo as idades faltantes por idade média na classe
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

#Removendo a coluna de cabines
train.drop('Cabin',axis=1,inplace=True)
train.dropna(inplace=True)

#Convertendo recursos categóricos
#  converter características categóricas em variáveis dummy usando pandas
sex = pd.get_dummies(train['Sex'],drop_first=True)  # drop_first=True > Para evitar a multi-colinaridade
embark = pd.get_dummies(train['Embarked'],drop_first=True)

train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)

#Construindo um modelo de regressão logísitica
# Divisão Treino-teste
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1),
                                                    train['Survived'], test_size=0.30,
                                                    random_state=101)

#Treino e predição
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

#Avalliação
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


