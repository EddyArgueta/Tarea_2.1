#Histograma y Mapa de Calor

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

# Usare un Histograma de todas las variables antes de usar otro metodo
df.hist(figsize=(15,8), bins=50, edgecolor='purple')
plt.tight_layout()
plt.show()

# Mapa de calor visualizando geografía y valor de casas
sb.scatterplot(data=df, x='longitude', y='latitude', hue='median_house_value', palette='coolwarm')
plt.title('Distribución geográfica vs. precio')
plt.show()



# Eliminamos columnas no numéricas si las hay
df = df.select_dtypes(include=["int64", "float64"])

# Separo variables dependientes e independientes
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]



# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear modelo y entrenar
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Evaluar precisión
train_datos = modelo.score(X_train, y_train)
test_datos = modelo.score(X_test, y_test)

print(f"entrenamiento : {train_datos}")
print(f"prueba : {test_datos}")