#Histograma y Mapa de Calor Segunda Parte

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

# Mostraremos las primeras filas del dataset
print("Primeras filas:")
print(df.head(30))

# Revisar valores únicos de la columna categórica
print(df['ocean_proximity'].value_counts())

# Información general del DataFrame
print(f"Información del DataFrame:")
print(df.info())

# Vemos las Estadísticas descriptivas
print(df.describe())

# Elimino filas con cualquier valor nulo
df = df.dropna()

# Estadísticas descriptivas para detectar datos limitados
#print(f"Resumen estadístico:")
#print(df.describe())

# Histograma 
df.hist(figsize=(15,8), bins=50, edgecolor='purple')
plt.tight_layout()
plt.show()

# Mapa de calor visualizando geografía y valor de casas
sb.scatterplot(data=df, x='longitude', y='latitude', hue='median_house_value', palette='coolwarm')
plt.title('Distribución geográfica vs. precio')
plt.show()

# Casos de ingresos extremadamente altos
sb.scatterplot(data=df[df['median_income'] > 14], x='longitude', y='latitude', hue='median_house_value', palette='coolwarm')
plt.title('Casas con ingreso > 14')
plt.show()



# Convertimos 'ocean_proximity' a variables dummy
dummies = pd.get_dummies(df['ocean_proximity'], dtype=int)
df = pd.concat([df, dummies], axis=1)
df.drop('ocean_proximity', axis=1, inplace=True)

# Eliminamos siempre las filas con valores nulos
df.dropna(inplace=True)
print(df.info())

# Mapa de calor de correlación
sb.set(rc={'figure.figsize':(15,8)})
sb.heatmap(df.corr(), annot=True, cmap='YlGnBu')
plt.title('Correlación entre variables')
plt.show()

# Ordenamos las correlaciones con respecto al precio
print(df.corr()['median_house_value'].sort_values(ascending=False))

# Se puede apreciar hasta aqui que los valores se encuentran mas altos entre 0.50 y -0.15
# Para: total_rooms, total_bedrooms, population, y households  

# Creo nuevas características
df['bedroom_ratio'] = df['total_rooms'] / df['total_bedrooms']

# Separamos las variables independientes y dependiente
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# Se Divide en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Se Entrenar el modelo sin escalar
modelo = LinearRegression()
modelo.fit(X_train, y_train)
predicciones = modelo.predict(X_test)

print("Resultados sin escalar:")
print(f"Resultados en entrenamiento: {modelo.score(X_train, y_train)}")
print(f"Resultados en prueba: {modelo.score(X_test, y_test)}")



#Calcular RMSE
rmse = mean_squared_error(y_test, predicciones)
print(f"RMSE sin escalar: {np.sqrt(rmse):.2f}")

# Visualizamos algunas relaciones
sb.histplot(df['total_rooms'])
plt.title('Distribución de total_rooms')
plt.show()

sb.scatterplot(data=df, x='median_house_value', y='median_income')
plt.title('Precio vs. Ingreso')
plt.show()

# Se muestran los resultados de:
# En el Histograma, las habitaciones entre 0 y 10000 tienen mayor distribucion
# Mientras que en el Scatterplot hay mayor influencia entre los 50000 y 400000 de valor contra 
# Un ingreso promedio de entre 0 a 8

# Escalamos las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenamos el modelo con datos escalados
modelo_escalado = LinearRegression()
modelo_escalado.fit(X_train_scaled, y_train)
predicciones_escaled = modelo_escalado.predict(X_test_scaled)

print("Resultados con escalado:")
print(f"Resultados en entrenamiento: {modelo_escalado.score(X_train_scaled, y_train)}")
print(f"Resultados en prueba: {modelo_escalado.score(X_test_scaled, y_test)}")

# 21. Calcular RMSE escalado
rmse_scaled = mean_squared_error(y_test, predicciones_escaled)
print(f"RMSE con escalado: {np.sqrt(rmse_scaled)}")

#Seguimos iguales
#Resultados con escalado:
#Resultados en entrenamiento: 0.6485002098617373
#Resultados en prueba: 0.6492437719008791
#RMSE con escalado: 69257.88641799583