import csv
import unicodedata
import sys

def limpiar_texto(texto):
    # Convierte a minúsculas
    texto = texto.lower()
    # Elimina acentos
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
    # Reemplaza espacios por guiones bajos
    texto = texto.replace(' ', '_')
    # Elimina comas
    texto = texto.replace(',', '')
    return texto

def limpiar_csv(archivo_entrada, archivo_salida):
    with open(archivo_entrada, newline='', encoding='utf-8') as f_in, \
         open(archivo_salida, 'w', newline='', encoding='utf-8') as f_out:
        lector = csv.reader(f_in)
        escritor = csv.writer(f_out)
        for fila in lector:
            fila_limpia = [limpiar_texto(celda) for celda in fila]
            escritor.writerow(fila_limpia)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python clean_csv.py archivo_entrada.csv archivo_salida.csv")
    else:
        limpiar_csv(sys.argv[1], sys.argv[2])