# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:04:36 2020

@author: willian mayrink
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

#Colocando na variavel cancer dados armazenados no proprio dataset da biblioteca sklearn.
cancer = load_breast_cancer()
cancer.keys() #ver quais chaves tem a variavel. Saida > dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names'])
print(cancer.DESCR) # Analisando a descrição dos dados.

#quantas caracteristicas contem e quais são elas ?
print( len(cancer['feature_names']) )
print(cancer.feature_names)

#transformando os dados de dataset do sklearn para dataframe do pandas.
df_cancer = pd.DataFrame(cancer.data,columns=cancer.feature_names) #1º paramentro os dados em si. 2º parametro os nomes de cada coluna dos respectivos dados do 1º parametro.
df_cancer['objetivo'] = pd.Series(cancer.target) #adicionando na coluna objetivo os dados de 0 ou 1 contido nos dados fornecidos.
df_cancer.head() #5 primeiras linhas da coluna cancer para uma pré analise dos dados.

#fazer um variavel com os dados de objetivos e suas contagens.
index=['malignos','benignos']
malignos= np.sum(df_cancer['objetivo'] == 0);
benignos= np.sum(df_cancer['objetivo'] == 1.0);
dados=[malignos, benignos]
target= pd.Series(dados, index=index)

X=df_cancer.iloc[:,:-1]  # variaveis independentes
y=df_cancer.iloc[:,-1]   # variavel dependente

#Separando os dados em dados de test e dados de treino. Default da funçao é 75% treino/ 25 % teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

#Criando o classificador knn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)

#Com o classificador implementando, vamos fazer a predição para um valor considerando a media de todas as caracteriscas.
media = df_cancer.mean()[:-1].values.reshape(1, -1)
predicao_media = knn.predict(media) #Retorno 1 que representa uma predição para paciente com resultado benigno.

#Predição com a amostra de teste.
predicao_teste= knn.predict(X_test)

#Calculando precisão (accuracy) de acerto dos dados para a amostra de testes
acc= knn.score(X_test, y_test) #91.61%


















