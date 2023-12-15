import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

sns.set_style()

df = pd.read_csv('datasets/heart-disease-clean.csv')

print(f'formato do df: {df.shape}')

print(df.head())

model = LogisticRegression()

X = df.drop('num', axis=1) # axis = 1 = dropa coluna
y = df['num']

X_train, X_test, y_train, y_test = train_test_split(X,y)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

model.fit(X_train,y_train)

X_test = scaler.transform(X_test)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

print(classification_report(y_test, y_pred))

fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, ax=ax)

ax.set_title("Matriz de Confusão")
ax.set_ylabel("Verdadeiro")
ax.set_xlabel("Previsto")

plt.show()



print("y_pred: \n", y_pred[0:5])
print("\ny_proba: \n", y_prob[0:5])