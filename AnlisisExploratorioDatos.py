#Importamos librerias necesarias 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

#----------------------------------------------

# Cargamos el dataset
df = pd.read_csv('apartments.csv', encoding='cp1252', low_memory=False)
# Visualizamos las primeras 5 filas
print(df.head())
# Filas y columnas del dataset
print(df.shape)
# Información del dataset
print(df.info())
# Estadísticas del dataset
print(df.describe())
# Ver cuántos valores nulos hay en cada columna
print(df.isnull().sum())
# Convertir columnas 'price_display' y 'square_feet' a numérico, manejando errores
df['price_display'] = pd.to_numeric(df['price_display'], errors='coerce')
df['square_feet'] = pd.to_numeric(df['square_feet'], errors='coerce')
# Eliminamos las filas con valores nulos o imputamos valores faltantes
df2 = df.dropna()
# Verificamos que se hayan eliminado los valores nulos
print(df2.isnull().sum())
print(df2.shape)

#------------------------DISTRIBUCIÓN DE VARIABLES CATEGORICAS---------------------------------
# Distribución de los valores para 'state'
print("Distribución de los valores para Estado")
print(df2['state'].value_counts())

print("Distribucion de los valores para Amenities")
print(df2['amenities'].value_counts())

#-------------------------------HISTOGRAMA PARA LA DISTRIBUCIÓN DE VARIABLE--------------------
# Histograma para la columna 'square_feet'
df2['square_feet'].hist(bins=20)
plt.title('Distribución del Área de los apartamentos')
plt.xlabel('Área (ft²)')
plt.ylabel('Frecuencia')
plt.show()

# Histograma para la columna 'bathrooms'
df2['bathrooms'].hist(bins=10)
plt.title('Distribución del Número de baños')
plt.xlabel('Número de baños')
plt.ylabel('Frecuencia')
plt.show()

#-------------------------------CORRELACIÓN DE VARIABLES NUMERICAS----------------------------
# Filtramos solo las columnas numéricas
correlation_matrix = df2.select_dtypes(include='number').corr()

# Visualizamos la matriz de correlación
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de correlación')
plt.show()

#----------------------IDENTIFICACIÓN DE VALORES ATIPICOS--------------------------------------
# Boxplot para 'square_feet'
df2.boxplot(column='square_feet')
plt.title('Boxplot del Área de los apartamentos')
plt.show()

# Boxplot para 'bathrooms'
df2.boxplot(column='bathrooms')
plt.title('Boxplot del Número de baños')
plt.show()

#----------------------------PREPROCESAMIENTO DE DATOS-------------------------------------------
# Imputamos valores nulos con la mediana para 'square_feet'
df2['square_feet'].fillna(df2['square_feet'].median(), inplace=True)

#---------------------------------ONE-HOT ENCODING-----------------------------------------------
# Codificación one-hot para la columna 'state'
df2 = pd.get_dummies(df2, columns=['state'], drop_first=True)

# Ver las columnas después de la codificación one-hot
print("Columnas del dataset después de one-hot encoding:")
print(df2.columns)

#Grafico de barras para la realacion entre 'Estado' y 'Area'
sns.barplot(x='state', y='square_feet', data=df)
plt.title('Relacion entre Estado y Area')
plt.show()

#--------------------IDENTIFICACIÓN DE COLUMNA IRRELEVANTES DE MANERA MANUAL------------------

# Eliminar columnas irrelevantes
df2.drop(columns=['title', 'body', 'amenities', 'currency', 'fee', 'has_photo', 
                  'address', 'source', 'cityname', 'category','pets_allowed','price_type'], inplace=True)

# Verificamos si aún hay datos nulos
print(df2.isnull().sum())

df2.dropna(inplace= True)
# Verificamos si aún hay datos nulos
print(df2.isnull().sum())

#----------------CORRELACION PARA DETECTAR CARACTERISTICAS ALTAMENTE CORRELACIONADAS------------------
#natruz de correlacion para detectar caracteristicas altamente correlacionadas
correlation_matrix = df2.select_dtypes(include='number').corr()

#Filtramos las caracteristicas altamente correlacionadas (mayor a 0.8)
highly_correlated = [column for column in correlation_matrix.columns if any(correlation_matrix[column] > 0.8)]
print("Caracteristicas altamente correlacionadas:")
print(highly_correlated)

#--------------SELECCION DE CARACTERISTICAS BASADAS EN LA IMPORTANCIA CON ARBOLES DE DESICIONES-----------

#  Definimos las variables predictoras y la variable objetivo
X = df2.drop('square_feet', axis=1)
Y = df2['square_feet']

#Se imprime todas las columnas que tengan valor objet (Categoricas)
categorical_columns = X.select_dtypes(include=['object']).columns
print("Colmnas tipo categoricas :",categorical_columns)

# Convertimos las variables categóricas en dummies
X_encoded = pd.get_dummies(X, drop_first=True)
#Imprimimos el DataFrame con las varibles dummy
print(X_encoded.head())

# Dividimos el dataset en entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, test_size=0.2, random_state=42)

# Aplicamos el modelo de Random Forest para regresión
model = RandomForestClassifier()  
model.fit(X_train, Y_train)

# Obtenemos la importancia de las características
feature_importances = model.feature_importances_
feature_names = X_encoded.columns

# Mostramos la importancia de las características
for feature, importance in zip(feature_names, feature_importances):
    print(f'{feature}: {importance}')


#-----------------------SELECCIÓN DE CARACTERISTICAS RFE----------------

# Escalamos las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definimos el modelo base de regresión logística
model2 = LogisticRegression(max_iter=10000)

# Aplicamos la selección de características con RFE
rfe = RFE(model2, n_features_to_select=5)
rfe.fit(X_train_scaled, Y_train)

# Obtener las características seleccionadas
selected_features = X_encoded.columns[rfe.support_]

# Mostrar las características seleccionadas
print("Características seleccionadas por RFE:")
print(selected_features)

#---------------------METODO DE ELIMINACIÓN BASADA EN LA VARIANZA---------------

#Aplicar el umbral de varianza para eliminar caracteristicas con varianza baja
selector = VarianceThreshold(threshold=0.1)
X_train_selected = selector.fit(X_train)

#Obtener las caracteristicas seleccionadas
selected_features = X.columns[selector.get_support(indices=True)]
print("Caracteristicas seleccionadas basadas en la Varianza:")
print(selected_features)

#--------------------ANALIZAR COMPONENTES PRINCIPALES-----------------------------
# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualizar la varianza explicada
print("Varianza explicada por cada componente:", pca.explained_variance_ratio_)