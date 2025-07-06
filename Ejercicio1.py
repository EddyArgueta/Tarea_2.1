import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Vamos a empezar a cargar el archivo CSV
df = pd.read_csv("housing.csv")

# Mostraremos las primeras filas del dataset
print("Primeras filas:")
print(df.head())

# Información general del DataFrame
print(f"Información del DataFrame:")
print(df.info())

# Elimino filas con cualquier valor nulo
df = df.dropna()

# Estadísticas descriptivas para detectar datos limitados
print(f"Resumen estadístico:")
print(df.describe())


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

# Hasta este punto, mis resultados fueron:
# entrenamiento : 0.6360185727313741
# prueba : 0.640086568899372
