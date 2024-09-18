#Importamos libreria de Pandas y le designamos el la clave pd
import pandas as pd
import chardet 
from sklearn.preprocessing import MinMaxScaler

#Checamos la Codificación del DataSet
#with open('apartments.csv', 'rb') as f:
    #result = chardet.detect(f.read())
#print(result)

#Leemos el DataSet a traves de Pandas y se lo asignamos a la variable df
#El encoding pertenecer al Windows-1252 / El parametro Low_memory sirve para procesar datasets de datos tipos mixto
df = pd. read_csv('apartments.csv' , encoding='cp1252', low_memory=False)

# Mostrar las primeras filas del dataset para ver la estructura
print("Primeras filas del dataset:")
print(df.head())

# Verificar los tipos de datos y la cantidad de valores faltantes
print("Información general del dataset:")
print(df.info())

# Revisar las estadísticas generales del dataset
print("Estadísticas generales del dataset:")
print(df.describe())

# Verificar valores faltantes en cada columna
print("Valores faltantes por columna:")
print(df.isnull().sum())

# Ejemplo: Rellenar valores faltantes en la columna "amenities" con "No disponible"
#df['amenities'].fillna('No disponible', inplace=True)

#Opcion valida para versiones actuales de Pandas, el parametro inplace=True no es correcta
df['amenities'] = df['amenities'].fillna('No disponible')

# Eliminar filas con valores faltantes en columnas críticas como "price_display"
df = df.dropna(subset=['price_display'])

#Imprimimos los tipos de datos de las columnas
print("Tipos de datos de las columnas:")
print(df.dtypes)

# Convertir la columna 'price_display' y 'square_feet' a numérico, manejando errores
df['price_display'] = pd.to_numeric(df['price_display'], errors='coerce')
df['square_feet'] = pd.to_numeric(df['square_feet'], errors='coerce')


# Ejemplo de codificación one-hot de la columna 'state'
df = pd.get_dummies(df, columns=['state', 'currency'], drop_first=True)

# Detectar outliers usando el rango intercuartílico
Q1 = df['price_display'].quantile(0.25)
Q3 = df['price_display'].quantile(0.75)
IQR = Q3 - Q1
# Filtrar valores que están fuera de 1.5 veces el IQR
df = df[~((df['price_display'] < (Q1 - 1.5 * IQR)) | (df['price_display'] > (Q3 + 1.5 * IQR)))]

scaler = MinMaxScaler()
df[['price_display', 'square_feet']] = scaler.fit_transform(df[['price_display', 'square_feet']])

# Verificar los valores normalizados
print("Valores normalizados:")
print(df[['price_display', 'square_feet']])

# Convertir la columna 'time' a formato de fecha
df['time'] = pd.to_datetime(df['time'], unit='s')

# Extraer año, mes y día de la semana
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month
df['day_of_week'] = df['time'].dt.dayofweek

# Eliminar columnas irrelevantes
df.drop(['source'], axis=1, inplace=True)

# Verificar nuevamente la estructura final
print(df.info())

# Comprobar si hay valores faltantes después de los cambios
print(df.isnull().sum())