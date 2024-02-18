import pandas as pd 
import matplotlib.pyplot as plt


df2018 = pd.read_csv('datasets/2018VAERSVAX.csv', encoding="Windows-1252") # https://vaers.hhs.gov/eSubDownload/index.jsp?fn=2018VAERSVAX.csv
df2023 = pd.read_csv('datasets/2023VAERSVAX.csv', encoding="Windows-1252") # https://vaers.hhs.gov/eSubDownload/index.jsp?fn=2023VAERSVAX.csv

# print(df2018.head())

# print(f'registros {df2023.shape[0]}') # 128319
# print(f'variáveis {df2023.shape[1]}') # 8

colunasUteis = ['VAX_TYPE']

df2018 = df2018[colunasUteis]
df2023 = df2023[colunasUteis]

df2023['VAX_TYPE'] = df2023['VAX_TYPE'].replace('COVID19-2','COVID19')

# print(df2023.isnull().sum())

''' 
print(df2023['VAX_TYPE'].unique())
['COVID19' 'TDAP' 'UNK' 'FLUA4' 'SMALLMNK' 'VARZOS' 'FLUX' 'FLU4' 'HIBV'
 'HPV4' 'HPV9' 'HEPA' 'RV5' 'MNQ' 'DTAPIPV' 'DTAP' 'FLUC4' 'HEP' 'PNC20'
 'TD' 'DTAPIPVHIB' 'MMR' 'MMRV' 'MENB' 'PNC13' 'VARCEL' 'DTAPHEPBIP' 'RV1'
 'PPV' 'FLUN3' 'DTPPVHBHPB' 'FLUR4' 'IPV' 'TYP' 'YF' 'HEPAB' 'FLUN4' 'RVX'
 'PNC15' 'ADEN_4_7' 'ANTH' 'FLU3' 'FLUC3' 'SMALL' 'MEN' 'RAB' 'HPVX'
 'FLUR3' 'DT' 'PNC' 'JEV1' 'BCG' 'TTOX' 'MEA' 'MU' 'FLUA3' 'TBE' 'DF'
 'DTOX' 'DTP' 'RSV' 'CHOL']
'''


# print(df2023.groupby('VAX_TYPE').value_counts().sort_values(ascending=False))

totalAeCovid = df2023.loc[( df2023['VAX_TYPE'] == 'COVID19'  )]['VAX_TYPE'].count()

print(f'Total AE covid: {totalAeCovid}') # total AE covid = 77501

print(f'% AE covid em relação a todas as vacinas: {(totalAeCovid / df2023.shape[0])*100}') # total AE covid = 77501

df2018_counts = df2018.groupby('VAX_TYPE').value_counts().sort_values(ascending=False).reset_index(name='counts')
df2023_counts = df2023.groupby('VAX_TYPE').value_counts().sort_values(ascending=False).reset_index(name='counts')

print(df2023_counts.head())

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

axs[0].bar(df2018_counts['VAX_TYPE'], df2018_counts['counts'])
axs[1].bar(df2023_counts['VAX_TYPE'], df2023_counts['counts'])

plt.show()