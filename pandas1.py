# tutorial learning pandas

import pandas as pd 

def linha():
    print('********************************************************************')


# df1 = pd.DataFrame({'Yes': [123,32], 'No': [20,52]})
df2 = pd.DataFrame({'Bob': ['Gostei', 'Detestei','seila'], 'Kyle':['Legal','Chato','sem opiniao']},
                    index=['Positivas', 'Negativas','Neutra'])

# df3 = pd.DataFrame({'AA': ['Gostei', 32], 'Kyle':['Legal','Chato']})

# s1 = pd.Series(['A','B','C','D'],
# index=['Letra A', 'Letra B', 'Letra C', 'Letra D'])


# print(df1['Yes'])
# print(df2)
# print(s1)

# df1.to_csv('exemplo1.csv')
df2.to_csv('datasets/exemplo2.csv')
# df3.to_csv('exemplo3.csv')
# s1.to_csv('serie1.csv')

df1 = pd.read_csv('datasets/exemplo2.csv',index_col=0)

linha()
print(df1.head)
linha()
print(f'shape {df1.shape}') # nro de linhas, nro de colunas
linha()
print(df1.Bob) # retorna uma série
# print(df1['Bob']) # retorna uma série
linha()
print(df1['Bob']['Positivas'])  # ou ['Bob'][0]

linha()
print(df1.iloc[0]) # pega linha 0

linha()
print(df1.iloc[:, 0]) # pega coluna 0

linha()
print(df1.iloc[:3, 0]) # pega até a 3º linhas da coluna 0

linha()
print(df1.iloc[1:3, 0]) # pega da linha 1 a 3 da coluna 0

linha()
print(df1.iloc[[0,2], 0]) # pega apenas a linha 0 e 2  da coluna 0

linha()
print(df1.iloc[-2:]) # pega as 2 ultimas linhas 

linha()
print(df1.loc['Positivas', 'Bob']) # pega a linha Positivas da coluna Bob

linha()
print(df1.loc['Positivas']) # pega a linha Positivas


linha()
print(df1.loc[:, 'Kyle']) # pega todas linhas da coluna Kyle



dfwines = pd.read_csv('datasets/winemag-data-130k-v2.csv', index_col=0)
linha()
#print(dfwines.loc[(dfwines.country == 'Italy') & (dfwines.points >= 90)])
# pega todas as linhas q pais = italy e points >= 90
# & e, | ou

linha()
#print(dfwines.loc[dfwines.country.isin(['Italy', 'France'])])
# is in semelhante ao sql
# tem tambem outros como isnull, notnull (compara se NaN)

linha()
#print(dfwines.loc[dfwines.price.notnull()])
# tras apenas as linhas cujo preço é definido


#dfwines['country'] = 'Brazil' # update seta Brazil para todos os country


dfwines['index_backwards'] = range(len(dfwines), 0, -1)
# cria uma coluna com os indices de tras pra frente

print(dfwines.head)

linha()
print(dfwines.points.describe())
linha()
print(dfwines.country.describe())

linha()
print(dfwines.points.mean()) # média dos pontos

linha()
print(dfwines.country.unique()) # valores não repetidos de países

linha()
print(dfwines.country.value_counts()) # conta quantas vezes cada valor aparece

linha()
#print(dfwines.country + " - " + dfwines.region_1) 
# retorna uma série que é a concatenação de duas outras

dfwines_points_mean = dfwines.points.mean()
#seriesmapeada = dfwines.points.map(lambda p: p - dfwines_points_mean)
linha()
#print(seriesmapeada)

def remean_points(row):
    row.points = row.points - dfwines_points_mean
    return row

#seriesmapeada = dfwines.apply(remean_points, axis='columns') # aplica o mapa por linha, index se for por coluna

linha()
#print(seriesmapeada)


trp_sum = dfwines.description.map(lambda p: 'tropical' in p).sum()
frt_sum = dfwines.description.map(lambda p: 'fruity' in p).sum()
print(trp_sum)
print(frt_sum)

# Who are the most common wine reviewers in the dataset? Create a Series whose index is the taster_twitter_handle category from the dataset, and whose values count how many reviews each person wrote.
reviews_written = dfwines.groupby('taster_twitter_handle').title.count()

# What is the best wine I can buy for a given amount of money? Create a `Series` whose index is wine prices and whose values is the maximum number of points a wine costing that much was given in a review. Sort the values by price, ascending (so that `4.0` dollars is at the top and `3300.0` dollars is at the bottom).
best_rating_per_price = dfwines.groupby('price').points.max()

# What are the minimum and maximum prices for each `variety` of wine? Create a `DataFrame` whose index is the `variety` category from the dataset and whose values are the `min` and `max` values thereof.
price_extremes = dfwines.groupby('variety').price.agg(['min','max'])

# What are the most expensive wine varieties? Create a variable `sorted_varieties` containing a copy of the dataframe from the previous question where varieties are sorted in descending order based on minimum price, then on maximum price (to break ties).
sorted_varieties = price_extremes.sort_values(by=['min', 'max'], ascending=False)

# Create a `Series` whose index is reviewers and whose values is the average review score given out by that reviewer. Hint: you will need the `taster_name` and `points` columns.
reviewer_mean_ratings = dfwines.groupby('taster_name').points.mean()

# What combination of countries and varieties are most common? Create a Series whose index is a MultiIndexof {country, variety} pairs. For example, a pinot noir produced in the US should map to {"US", "Pinot Noir"}. Sort the values in the Series in descending order based on wine count.
country_variety_counts = dfwines.groupby(['country','variety']).title.count().sort_values(ascending=False)
print(country_variety_counts)



linha()
linha()
print(dfwines.index)







