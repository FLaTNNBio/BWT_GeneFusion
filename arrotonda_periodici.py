import pandas as pd

# Carica il CSV
df = pd.read_csv("full_dataset_vectors.csv", sep=";")

# Arrotonda tutte le colonne numeriche a 2 decimali
df_rounded = df.round(2)

# Salva il file arrotondato
df_rounded.to_csv("full_dataset_vectors_rounded.csv", index=False, sep=";")
print("✅ File salvato con valori arrotondati.")
