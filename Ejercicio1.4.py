# Intentando mejorar mas...

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar datos
df = pd.read_csv('./housing.csv')

# Eliminar nulos
df.dropna(inplace=True)

# Crear nuevas columnas
df['bedroom_ratio'] = df['total_rooms'] / df['total_bedrooms']
df['rooms_per_person'] = df['total_rooms'] / df['population']
df['is_max_age'] = (df['housing_median_age'] == 50).astype(int)
df['log_income'] = np.log(df['median_income'] + 1)  

# Dummies para ocean_proximity
dummies = pd.get_dummies(df['ocean_proximity'], dtype=int)
df = pd.concat([df.drop('ocean_proximity', axis=1), dummies], axis=1)

# Variables a usar
features = ['median_income', 'housing_median_age', 'bedroom_ratio', 'rooms_per_person',
            'is_max_age', 'log_income', 'NEAR BAY', 'INLAND']

X = df[features]
y = df['median_house_value']

# Escalamiento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Generar variables polinomiales
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# División (dejamos solo 5% para prueba)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.05, random_state=42)

# Modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Predicciones
pred_train = modelo.predict(X_train)
pred_test = modelo.predict(X_test)

# Evaluaciones
score_train = modelo.score(X_train, y_train)
score_test = modelo.score(X_test, y_test)
rmse = np.sqrt(mean_squared_error(y_test, pred_test))


print(f'Resultado de entrenamiento: {score_train}')
print(f'Resultado de prueba: {score_test}')
print(f'RMSE: {rmse}')

# Gráfica de errores
plt.figure(figsize=(10,5))
errores = y_test - modelo.predict(X_test)
sb.histplot(errores, kde=True, color='purple')
plt.title("Distribución de errores (y_test - predicción)")
plt.xlabel("Error")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.tight_layout()
plt.show()