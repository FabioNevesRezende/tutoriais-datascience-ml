import matplotlib.pyplot as plt
import pandas as pd


# df = pd.read_csv('datasets/emendas-utf8.csv',encoding="UTF-8",engine='python', thousands='.', decimal=",",sep=';')

# df_udi = df.loc[(df['nome_ente'] == 'Uberlândia')]

df_udi = pd.read_csv('datasets/emendas-udi.csv',encoding="UTF-8",engine='python', decimal=".",sep=',', parse_dates=True)

# print('************************************')
# print('Tipos de dados')
# print(df.dtypes)

print('************************************')
print('Categoria especial de despesas')
print(df_udi.cat_economica_despesa.unique())


# print('************************************')
# print(f'linhas: {df_udi.shape[0]}') 
# print(f'colunas: {df_udi.shape[1]}') 

# print('************************************')
# print('favorecidos uberlândia')
# print(df_udi.nome_favorecido.unique())


print('************************************')
print('valor por favorecido uberlândia')

dfgr = df_udi.sort_values(by=['ano']).groupby(['nome_favorecido','ano']).valor.sum()

print(
    dfgr.describe()
)

dfgr.hist(bins=15, figsize=(15,10))

plt.show()

#df_udi.to_csv('emendas-udi.csv')