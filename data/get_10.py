import pandas as pd

# Leer el archivo CSV
df = pd.read_csv('sample_ds3.csv')

# Obtener 10 registros aleatorios
aleatorios_10 = df.sample(n=32)

# Mostrar los resultados
aleatorios_10.to_csv('../exact_test_DS3.csv', index=False)