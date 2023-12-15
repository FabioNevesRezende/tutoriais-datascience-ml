import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('ocorrencias_aviacao.csv', parse_dates=True,index_col='codigo_ocorrencia')

print(df.loc[201305055424986,["ocorrencia_latitude", "ocorrencia_longitude"]])

fig, ax = plt.subplots(figsize=(12,8))

df.plot.scatter(x='ocorrencia_longitude', y='ocorrencia_latitude', s=1, alpha=0.4, ax=ax)

plt.tight_layout()
plt.show()
