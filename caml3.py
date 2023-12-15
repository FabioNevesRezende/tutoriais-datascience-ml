import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()

df = pd.read_csv('CAML3.SA.csv', parse_dates=True,index_col='Date')

print(df.index)
print(df.info)

df.High.plot();

plt.show()

df.Volume.hist()
plt.show()

dfnorm = (df - df.mean()) / df.std()


print(dfnorm.head())