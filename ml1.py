# regressão linear simples e múltipla

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

sns.set()

df = pd.read_csv('precificacao_housing_plus.csv')

print(df.head())

X = df['GrLivArea'].values.reshape(-1,1)
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

fix, ax = plt.subplots(figsize=(8,5))

df.plot.scatter(x='GrLivArea', y='SalePrice', ax=ax)


# plt.tight_layout()
# plt.show()

lr_model = LinearRegression()
lr_model.fit(X_train,y_train)

print("Coeficiente:\t", lr_model.coef_)
print("Intercepto:\t", lr_model.intercept_)

fig, ax = plt.subplots()
ax.scatter(X, y, s=1, color='blue')
ax.plot(X, (lr_model.coef_ * X + lr_model.intercept_), '--r', linewidth=1)

plt.tight_layout()
plt.show()

y_pred = lr_model.predict(X_test)

print(f'R2 Score {r2_score(y_test,y_pred)}')
print(f'Mean absolute error {mean_absolute_error(y_test,y_pred)}')
print(f'Mean square error {mean_squared_error(y_test,y_pred)}')

df.drop('Id', axis=1, inplace=True)

X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# dividir o dataset entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y)

# criar e treinar um modelo de Regressão Linear
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print('************************************')
print("Coeficiente:\t", lr_model.coef_)
print("Intercepto:\t", lr_model.intercept_)

y_pred = lr_model.predict(X_test)

# verificar desempenho do modelo
print("R2 Score:\t", r2_score(y_test, y_pred))
print("MAE:\t\t", mean_absolute_error(y_test, y_pred))
print("MSE:\t\t", mean_squared_error(y_test, y_pred))