import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('listings.csv',index_col=0)

print('************************************')
print(f'entradas: {df.shape[0]}') 
print(f'variáveis: {df.shape[1]}') 

#print(df.dtypes) #data types


print('************************************')
print(
    (df.isnull().sum() # qtdade regs nulos pelo nro de linhas = % de nulos
    /
    df.shape[0]).sort_values(ascending=False) # em ordem decrescente
)


# df.hist(bins=15, figsize=(15,10))

# plt.show()

print('************************************')
print(
    df[['price','minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count','availability_365']].describe()
)

dfteste = df[['price','minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count','availability_365','latitude','longitude']]
# cria um novo df contendo apenas as colunas citadas

dfteste = dfteste.loc[(dfteste.price <= 1000) & (dfteste.minimum_nights <= 10)]


print('************************************')
print('nova descrição do dfteste:')
print(
    dfteste.describe()
)

# boxplot da colina minimum_nights
# df.minimum_nights.plot(kind='box', vert=False, figsize=(15,3))
# plt.show()

print('************************************')
print('minimum_nights, valores acima de 30:')
print(f'{len(df[df.minimum_nights > 30])} entradas')
print(f'{ ((len(df[df.minimum_nights > 30])) / df.shape[0])*100 }%')

# dfteste.hist(bins=15, figsize=(15,10))

# plt.show()

dftestecorr = dfteste.corr()

print('************************************')
print(dftestecorr)

# sns.heatmap(dftestecorr, cmap='RdBu', fmt='.2f', square=True, linecolor='white', annot=True)
# plt.show()


print('************************************')
print('tipo de imovel mais alugado')
print(df.room_type.value_counts())
print(f'{(df.room_type.value_counts() / df.shape[0]) * 100}')

print('************************************')
print('localidade mais cara no RJ')
print(
    df.groupby(['neighbourhood']).price.mean().sort_values(ascending=False)[:20]
)


print('************************************')

dfteste.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, c=dfteste['price'], s=8, cmap=plt.get_cmap('jet'), figsize=(12,8))
plt.show()

# desvio padrão preços:
print('************************************')
print('desvio padrão preços')
print(f'{dfteste.price.std()}')

# maior dos preços:
print('************************************')
print('maior dos preços')
print(f'{dfteste.price.max()}')
# menor preços:
print('************************************')
print('menor preços')
print(f'{dfteste.price.min()}')