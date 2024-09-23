import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


#Cargamos el dataset
df= pd.read_csv('apartments.csv', encoding='cp1252', low_memory=False)

#Aqui si se pueden visualizar las primeras 5 filas
print(df.head())

#Filas y columnas del dataset
print(df.shape)

#Informacion del dataset es decir el tipo de dato de cada columna y si hay valores nulos
print(df.info())

#Estadisticas del dataset
print(df.describe())

#Ver cuantos valores nulos hay en cada columna
print(df.isnull().sum())

# Convertir la columna 'price_display' y 'square_feet' a numérico, manejando errores
df['price_display'] = pd.to_numeric(df['price_display'], errors='coerce')
df['square_feet'] = pd.to_numeric(df['square_feet'], errors='coerce')

#Podemos eliminar las filas con valores nulos o imputar los valores faltantes
#En este caso crearemos una copia del dataset sin los valores nulos
df2 = df.dropna()
#Verificamos que se hayan eliminado los valores nulos
print(df2.isnull().sum())
#Sin embargo el data set se redujo a mas la mitad en filas
print(df2.shape)

print("Distribucion de los valores para Estado")
print(df2['state'].value_counts())


print("Distribucion de los valores para Amenities")
print(df2['amenities'].value_counts())


#Histograma para la columna Area
df2['square_feet'].hist(bins=20)
plt.title('Distribucion del Area de los apartamentos')
plt.xlabel('Area (m2)')
plt.ylabel('Frecuencia')
plt.show()

#Histograma para la columna de Baños
df2['bathrooms'].hist(bins=10)
plt.title('Distribucion del Numero de baños')
plt.xlabel('Numero de baños')
plt.ylabel('Frecuencia')
plt.show()

#Filtramos solo las columnas numericas
correlation_matrix = df2.select_dtypes(include='number').corr()

#Ahora visualizamos la matriz de correlacion
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de correlacion')
plt.show()


df2.boxplot(column='square_feet')
plt.title('Boxplot del Area de los apartamentos')
plt.show()

#Boxplot para la columna baños
df2.boxplot(column='bathrooms')
plt.title('Boxplot del Numero de baños')
plt.show()

#Imputamos valores nulos con la mediana 
df2['square_feet'].fillna(df2['square_feet'].median(), inplace=True)


#APlicamos tecnica one hot encoding para convertir las variables categoricas en numericas
# Ejemplo de codificación one-hot de la columna 'state'
df2 = pd.get_dummies(df, columns=['state'], drop_first=True)

# Ahora veremos que el dataset elimino las columnas 'state' y agrego nuevas columnas con la codificación one-hot
print("Columnas del dataset:")
print(df2.columns)

#Grafico de barras para la realacion entre 'Estado' y 'Area'
sns.barplot(x='state', y='square_feet', data=df)
plt.title('Relacion entre Estado y Area')
plt.show()

#ELiminar columnas irrelevantes
df2.drop(columns=['title','body','amenities', 'currency','fee','has_photo','address','source','cityname','category','pets_allowed'], inplace=True)

#one hot encoding para la columna 'pets_allowed' y 'price_type'
df2 = pd.get_dummies(df2, columns=['price_type'], drop_first=True)
# Ahora veremos que el dataset elimino las columnas 'state' y agrego nuevas columnas con la codificación one-hot
print("Columnas del dataset:")
print(df2.columns)

#Ver datos nulos de df2
print(df2.isnull().sum())

#Realizar limpieda de datos nulos 
df2.dropna(inplace=True)
#Ver datos nulos de df2
print(df2.isnull().sum())

#natruz de correlacion para detectar caracteristicas altamente correlacionadas
correlation_matrix = df2.select_dtypes(include='number').corr()

#Filtramos las caracteristicas altamente correlacionadas (mayor a 0.8)
highly_correlated = [column for column in correlation_matrix.columns if any(correlation_matrix[column] > 0.8)]
print("Caracteristicas altamente correlacionadas:")
print(highly_correlated)

#Definimos las variables predictoras y la variable objetivo
X = df2.drop('square_feet',axis=1)
Y = df2['square_feet']

categorical_columns = X.select_dtypes(include=['object']).columns
print(categorical_columns)

X_encoded = pd.get_dummies(X, drop_first=True)  # Convierte variables categóricas en dummies
print(X_encoded.head())


# Dividir el dataset en entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, test_size=0.2, random_state=42)

#Aplicamos el modelo de Random Forest
model = RandomForestRegressor(n_estimators=50, max_depth=10)
model.fit(X_train, Y_train)

#Obtenemos la importancia de las caracteristicas
feature_importances = model.feature_importances_
feature_names = X_encoded.columns 

#Mostramos la importancia de las caracteristicas
for feature, importance in zip(feature_names, feature_importances):
    print(f'{feature}: {importance}')


# Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir el modelo base de regresión logística
model2 = LogisticRegression(solver='liblinear', max_iter=1000)

# Aplicar la selección de características con RFE
rfe = RFE(model2, n_features_to_select=5)  # Seleccionar las 5 mejores características
rfe.fit(X_train_scaled, Y_train)

# Obtener las características seleccionadas
selected_features = X_encoded.columns[rfe.support_]

# Mostrar las características seleccionadas
print("Características seleccionadas por RFE:")
print(selected_features)

# Evaluar el modelo con las características seleccionadas
model2.fit(X_train_scaled[:, rfe.support_], Y_train)
accuracy = model2.score(X_test_scaled[:, rfe.support_], Y_test)
print(f"Precisión del modelo con las características seleccionadas: {accuracy:.2f}")