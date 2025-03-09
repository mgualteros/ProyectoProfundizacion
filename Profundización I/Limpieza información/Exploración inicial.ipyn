import pandas as pd 
import numpy as np
import pandas as pd
df = pd.read_excel('C:\\Users\\USER\\OneDrive\\Escritorio\\Maestría\\ProyectoProfundizacion\\Profundización I\\Base de Datos\\casos.xlsx')
df.info()
casos = df['Descripción']
longitudes = casos.str.len()
summarylongitudes = longitudes.describe()
print(summarylongitudes)

casos_limpios = casos.str.replace(r'\s+', ' ', regex=True)  # Eliminar espacios extra y saltos de línea
casos_limpios = casos_limpios.str.strip()  # Eliminar espacios al inicio y final
casos_limpios = casos_limpios.str.lower()  # Convertir todo a minúsculas

longitudes2 = casos_limpios.str.len()
summarylongitudes2 = longitudes2.describe()
print(summarylongitudes2)

print(casos_limpios[2])

