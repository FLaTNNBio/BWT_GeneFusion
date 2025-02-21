#!/usr/bin/env python
"""
Crea vettori numerici per ogni sequenza a partire dal file CSV dei B-mers.
Il file CSV di input deve contenere le colonne:
  ID, b-mer, frequency, label
Per ogni sequenza (ID), il vettore avrà come colonne i B-mers e il valore sarà la frequenza con cui appare.
Salva il risultato in un nuovo file CSV.

Usage:
    python crea_vettori.py input_bmers.csv -o output_vectors.csv
"""

import argparse
import pandas as pd

def crea_vettori(input_csv, output_csv):
    # Legge il file CSV
    df = pd.read_csv(input_csv, sep=";")
    
    # Normalizza i nomi delle colonne (in minuscolo e rimuove eventuali spazi)
    df.columns = df.columns.str.strip().str.lower()
    # Se le colonne sono: id, b-mer, frequency, label, va bene.
    
    # Crea la pivot table: index = ID e label, columns = b-mer, values = frequency, fill missing with 0
    df_pivot = df.pivot_table(index=["id", "label"], columns="b-mer", values="frequency", fill_value=0)
    
    # Ripristina l'indice in colonne
    df_vettori = df_pivot.reset_index()
    
    # Salva il risultato in CSV
    df_vettori.to_csv(output_csv, index=False, sep=";")
    print(f"✅ Vettori creati e salvati in {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crea vettori numerici per ogni sequenza a partire dal file CSV dei B-mers."
    )
    parser.add_argument("input_csv", help="File CSV con i B-mers (colonne: ID, b-mer, frequency, label).")
    parser.add_argument("-o", "--output_csv", required=True, help="Nome del file CSV di output con i vettori.")
    args = parser.parse_args()
    
    crea_vettori(args.input_csv, args.output_csv)
