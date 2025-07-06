#Nuevo Intento cambiando variables

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

scaler = StandardScaler()

# Vamos a empezar a cargar el archivo CSV
df = pd.read_csv("housing.csv")

df_model = df.copy()

# Variables nuevas
df_model['bedroom_ratio'] = df_model['total_rooms'] / df_model['total_bedrooms']
df_model['log_income'] = np.log1p(df_model['median_income'])
df_model['is_max_age'] = df_model['housing_median_age'].apply(lambda x: 1 if x == 52 else 0)

# Eliminano columnas innecesarias
df_model.drop(['total_rooms', 'total_bedrooms', 'median_income', 
'longitude', 'latitude', 'population', 'households'], axis=1, inplace=True)

# Dummy variables
df_model = pd.concat([df_model, pd.get_dummies(df_model['ocean_proximity'], dtype=int)], axis=1)
df_model.drop('ocean_proximity', axis=1, inplace=True)

# Se siguen limpiando nulos
df_model.dropna(inplace=True)

# Separar
X = df_model.drop('median_house_value', axis=1)
y = df_model['median_house_value']

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modelo lineal
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Evaluar
print(f"Resultados de entrenamiento: {modelo.score(X_train, y_train)}")
print(f"Resultados de prueba: {modelo.score(X_test, y_test)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, modelo.predict(X_test)))}")

# regplot tira una linea roja cruzando de lado a lado, 
# y lo mas cercano es lo mas certero
sb.regplot(x='log_income', y='median_house_value', data=df_model, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('Relación entre log(ingreso) y precio de casa')
