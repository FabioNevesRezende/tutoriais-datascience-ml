import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv("datasets/heart-disease-clean.csv")

# print("Tamanho de df: {}\n".format(df.shape))

# print(df.head(10))

model = DecisionTreeClassifier(max_depth=4, criterion="entropy")
model_xg = XGBClassifier(
    learning_rate=0.05,
    n_estimators=100,
    max_depth=4,
    subsample = 0.9,
    colsample_bytree = 0.1,
    gamma=1,
    random_state=42
)

X = df.drop('num', axis=1)
y = df['num']

X_train, X_test, y_train, y_test = train_test_split(X,y)

model.fit(X_train,y_train)
model_xg.fit(X_train,y_train)

y_pred = model.predict(X_test)
y_pred_xg = model_xg.predict(X_test)

print(classification_report(y_test,y_pred))


fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, ax=ax)

ax.set_title("Matriz de confusão Decision Tree")
ax.set_ylabel("Verdadeiro")
ax.set_xlabel("Previsto")

plt.show()

print("accuracy_score")
print(accuracy_score(y_test,y_pred))
print("accuracy_score xgboost")
print(accuracy_score(y_test,y_pred_xg))