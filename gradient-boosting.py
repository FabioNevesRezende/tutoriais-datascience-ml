from xgboost import XGBRegressor
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

# Read the data
data = pd.read_csv('datasets/melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price

# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

my_model = XGBRegressor( early_stopping_rounds=5, n_estimators=1000, learning_rate=0.05, n_jobs=4) 
# n_estimators, nro de iterações do modelo, qts vezes declarar novo modelo e refazer predições
# n_jobs, nro de threads para paralelismo e acelerar aprendizado
# learning_rate, multiplica as previsões dos modelos a cada iteração, faz com que novas iterações ajudem menos melhorando no overfitting
my_model.fit(X_train, y_train,eval_set=[(X_valid, y_valid)], verbose=False) # early_stopping_rounds, para as iterações quando não há mais ganho na melhoria do modelo


predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
