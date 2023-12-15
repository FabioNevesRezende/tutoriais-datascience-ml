import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

sns.set()

df = pd.read_csv('ar-cond.csv')

print(df.head())

X = df[['local_temperature','humudity']]
y = df['cond_temperature']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

fig, ax = plt.subplots(figsize=(8,4))
ax = fig.add_subplot(projection='3d')

ax.scatter(df['local_temperature'],df['humudity'],df['cond_temperature'])

plt.tight_layout()
plt.show()

lr_model = LinearRegression()
lr_model.fit(X_train,y_train)

print("Coeficiente:\t", lr_model.coef_)
print("Intercepto:\t", lr_model.intercept_)

y_pred = lr_model.predict(X_test)
# y_pred2 = lr_model.predict([[40,5],[37,5]])

# print(y_pred2)

print(f'R2 Score {r2_score(y_test,y_pred)}')
print(f'Mean absolute error {mean_absolute_error(y_test,y_pred)}')
print(f'Mean square error {mean_squared_error(y_test,y_pred)}')




