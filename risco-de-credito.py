import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

sns.set()


df = pd.read_csv('datasets/acquisition_train.csv')

print('************************************')
print("Variáveis e entradas:")
dflinhas = df.shape[0]
dfcolunas = df.shape[1]
print(f'Entradas: {dflinhas}')
print(f'Variáveis: {dfcolunas}')


print('************************************')
print('info do dataset:')
dfinfo = df.describe()
print(dfinfo)

print('************************************')
print('Nulos:')
print(df.isnull().sum().sort_values(ascending=False))
print('Nulos %')
print((df.isnull().sum() / dflinhas).sort_values(ascending=False))


print('************************************')
print('Valores únicos:')
print(df.nunique().sort_values())


print('************************************')
print('Proporção entre inadimplentes do dataset')

print(df['target_default'].value_counts() / df.shape[0])

print('************************************')
print('limpeza de dados nulos e colunas inúteis ao processo de machine learning')

df_clean = df.copy()

df_clean['reported_income'] = df_clean['reported_income'].replace(np.inf, np.nan)

df_clean.drop(['ids','target_fraud','external_data_provider_credit_checks_last_2_year','channel'], axis=1, inplace=True)

df_clean.loc[df_clean['external_data_provider_email_seen_before'] == -999, 'external_data_provider_email_seen_before'] = np.nan

df_clean.drop(labels=['reason', 'zip', 'job_name', 'external_data_provider_first_name',
            'lat_lon', 'shipping_zip_code', 'user_agent', 'profile_tags',
            'application_time_applied', 'email', 'marketing_channel',
            'profile_phone_number', 'shipping_state'], axis=1, inplace=True)

df_clean.dropna(subset=['target_default'], inplace=True)

num_df = df_clean.select_dtypes(exclude='object').columns # colunas numéricas
cat_df = df_clean.select_dtypes(include='object').columns # colunas categóricas

df_clean.last_amount_borrowed.fillna(value=0, inplace=True)
df_clean.last_borrowed_in_months.fillna(value=0, inplace=True)
df_clean.n_issues.fillna(value=0, inplace=True)

imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(df_clean.loc[:,num_df])
df_clean.loc[:,num_df] = imputer.transform(df_clean.loc[:,num_df]) # substitui os NaN pela mediana da coluna

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(df_clean.loc[:,cat_df])
df_clean.loc[:,cat_df] = imputer.transform(df_clean.loc[:,cat_df]) # substitui os NaN pelo valor mais frequente na coluna (moda)

print(df_clean.isnull().sum())



print('************************************')
print('Normaliza o dataset')

scaled_df = df_clean.copy()

num_cols = scaled_df.drop('target_default',axis=1).select_dtypes(exclude='object').columns

scaled_df[num_cols] = StandardScaler().fit_transform(scaled_df[num_cols].values)


print('************************************')
print('Transforma variaveis categóricas em numéricas')

encoded_df = scaled_df.copy()
# extrair as colunas categóricas
cat_cols = encoded_df.select_dtypes('object').columns

# codificar cada coluna categórica
for col in cat_cols:
  encoded_df[col+'_encoded'] = LabelEncoder().fit_transform(encoded_df[col])
  encoded_df.drop(col, axis=1, inplace=True)

print('************************************')
print('Prepara base de treino e teste')

X = encoded_df.drop('target_default', axis=1).select_dtypes(exclude='object')
y = encoded_df['target_default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


print('************************************')
print('Cria e treina o modelo usando o algoritmo XGBoost classifier e o avalia usando o accuracy_score')

ml_model = XGBClassifier(learning_rate=0.01, n_estimators=1000,
                         max_depth=3, subsample = 0.9,
                         colsample_bytree = 0.1, gamma=1,
                         random_state=42)
ml_model.fit(X_train, y_train)

y_pred = ml_model.predict(X_test)

# ver performance do algoritmo
print('************************************')
print("\nAccuracy Score:")
print (accuracy_score(y_test, y_pred))