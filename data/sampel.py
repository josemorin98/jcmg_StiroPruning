import pandas as pd

# Ruta del archivo CSV original
csv_path = './training_vectors.csv'


# Cargar el DataFrame
df = pd.read_csv(csv_path)

# Tomar una muestra aleatoria del 25%
sample_df = df.sample(frac=0.01, random_state=42)  # Puedes cambiar el random_state si quieres variación

print(sample_df.shape)

# Guardar la muestra en un nuevo archivo (opcional)
sample_df.to_csv('sample.csv', index=False, encoding='utf-8')


print(f"Muestra generada con {len(sample_df)} filas (5% del total).")
