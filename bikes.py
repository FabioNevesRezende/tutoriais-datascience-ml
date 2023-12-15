import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('df_bike_rides.csv', parse_dates=True)
print('shape:')

print('************************************')
print(f'entradas: {df.shape[0]}') 
print(f'variaveis: {df.shape[1]}') 

print('Total e % de valores ausentes:')
print(df.isnull().sum())

print((df.isnull().sum() / df.shape[0])*100)

print('Exluindo valores ausentes')

df_row_dropna = df.dropna(subset=['user_gender'], axis=0) # axis = 0, linha, axis=1, coluna, exclui todas as linhas com user_gender null

print('************************************')
print(f'entradas: {df_row_dropna.shape[0]}') 
print(f'variaveis: {df_row_dropna.shape[1]}') 

df_col_dropna = df.dropna(axis=1) # exclui todas colunas q tem pelo menos 1 nulo

print('************************************')
print(f'entradas: {df_col_dropna.shape[0]}') 
print(f'variaveis: {df_col_dropna.shape[1]}') 

print('Preenchendo valores ausentes')

df.ride_duration.fillna(df.ride_duration.median(), inplace=True)

df.user_residence.fillna('DF', inplace=True)

print(f'ride_duration null: {df.ride_duration.isnull().sum()}') 

print(f'user_residence null: {df.user_residence.isnull().sum()}') 

print(f'user_residences:') 
print(df.user_residence.unique())


