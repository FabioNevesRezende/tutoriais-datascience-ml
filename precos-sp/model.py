import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from joblib import dump

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# importar o dataset para um dataframe
df = pd.read_csv('sao-paulo-properties-april-2019.csv')

df['District'] = df['District'].apply(lambda x: x.split('/')[0])

# ver as 5 primeiras entradas
print(df.head())

# dummy variables
df = pd.get_dummies(df)

# separar entre variáveis X e y
X_simp = df[['Condo','Size','Rooms','Suites']]
y_simp = df['Price']

# split entre datasets de treino e teste
X_train_simp, X_test_simp, y_train_simp, y_test_simp = train_test_split(X_simp, y_simp, test_size=0.33)

# instanciar e treinar o modelo
model = RandomForestRegressor(random_state=42)
model.fit(X_train_simp, y_train_simp)

# fazer as previsões em cima do dataset de teste
y_pred_simp = model.predict(X_test_simp)

# métricas de avaliação
print("r2: \t{:.4f}".format(r2_score(y_test_simp, y_pred_simp)))
print("MAE: \t{:.4f}".format(mean_absolute_error(y_test_simp, y_pred_simp)))
print("MSE: \t{:.4f}".format(mean_squared_error(y_test_simp, y_pred_simp)))

# exportar o modelo
dump(model, 'models/model.joblib') 
dump(X_train_simp.columns.values, 'models/features.names') 
