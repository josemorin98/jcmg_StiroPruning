import pandas as pd

# Leer el archivo CSV
df = pd.read_csv('sample_ds3.csv')

# Obtener los primeros 10 registros
primeros_10 = df.head(17)

# Mostrar los resultados
primeros_10.to_csv('../exact_test_DS3.csv', index=False)