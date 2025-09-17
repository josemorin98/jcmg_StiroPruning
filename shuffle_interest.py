import pandas as pd
import random

# Mezcla aleatoriamente la columna 'interest' de un CSV y guarda el resultado en otro archivo

def shuffle_interest_column(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    if 'interest' not in df.columns:
        raise ValueError("La columna 'interest' no existe en el archivo.")
    interests = df['interest'].tolist()
    random.shuffle(interests)
    df['interest'] = interests
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    input_csv = "no_exact_test_DS3.csv"
    output_csv = "no_exact_test_DS3_shuffled.csv"
    shuffle_interest_column(input_csv, output_csv)
