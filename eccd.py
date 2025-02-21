#!/usr/bin/env python
"""
Riduce la dimensionalità del dataset mediante ECCD:
Calcola l'entropia globale (H(Y)) e l'entropia condizionata per ciascuna feature (B-mer),
poi rimuove le feature con ECCD inferiore a una soglia.
Genera un nuovo file CSV con le stesse colonne (ID, Label, ... B-mers selezionati).

Usage:
    python riduci_dataset.py full_dataset_vectors.csv -o full_dataset_vectors_reduced.csv --threshold 0.02
"""

import argparse
import numpy as np
import pandas as pd

def entropy(labels):
    """
    Calcola l'entropia di Shannon per una serie di etichette.
    H(Y) = - sum_i P(y_i) log2(P(y_i))
    """
    counts = labels.value_counts(normalize=True)
    return -np.sum(counts * np.log2(counts))

def conditional_entropy(feature_values, labels):
    """
    Calcola l'entropia condizionata H(Y|X) per una feature.
    
    feature_values: Serie Pandas (1-D) contenente i valori della feature per ogni sequenza.
    labels: Serie Pandas contenente le etichette corrispondenti (es. 1 = fusion, 0 = no fusion).
    """
    total = len(feature_values)
    # Appiattiamo i valori in un array 1D e li convertiamo in numerico
    feature_values = pd.to_numeric(pd.Series(feature_values.values.ravel()), errors='coerce')
    
    present = feature_values > 0
    absent = ~present

    p_present = present.sum() / total
    p_absent = absent.sum() / total

    def entropy_subset(subset):
        if len(subset) == 0:
            return 0
        probs = subset.value_counts(normalize=True)
        return -np.sum(probs * np.log2(probs))
    
    H_present = entropy_subset(labels[present]) if present.sum() > 0 else 0
    H_absent = entropy_subset(labels[absent]) if absent.sum() > 0 else 0

    cond_ent = p_present * H_present + p_absent * H_absent
    return cond_ent

def process_file(input_csv, output_csv, threshold):
    """
    Legge il file CSV con i vettori numerici (ID, Label, ... B-mer frequenze),
    calcola per ogni feature il valore ECCD = H(Y) - H(Y|X),
    e mantiene solo le feature con ECCD >= threshold.
    Salva il nuovo dataset ridotto in un file CSV.
    """
    # Legge il CSV (assumendo che il separatore sia ";")
    df = pd.read_csv(input_csv, sep=";", engine="python", encoding="utf-8")
    print("Shape iniziale del dataset:", df.shape)
    
    # Normalizza i nomi delle colonne (minuscolo, rimuove spazi)
    df.columns = df.columns.str.strip().str.lower()
    
    if "id" not in df.columns or "label" not in df.columns:
        raise KeyError("Il file CSV deve contenere le colonne 'ID' e 'Label' (case-insensitive).")
    
    # Converte le colonne numeriche (dalla terza in poi) in numerico
    df.iloc[:, 2:] = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
    
    # Calcola l'entropia globale H(Y)
    global_ent = entropy(df["label"])
    print(f"✅ Entropia globale (H(Y)) = {global_ent:.4f}")
    
    # Calcola ECCD per ogni feature (B-mer)
    feature_columns = [col for col in df.columns if col not in ["id", "label"]]
    selected_features = []
    
    for col in feature_columns:
        if len(df[col]) != len(df["label"]):
            print(f"Warning: La colonna {col} ha lunghezza {len(df[col])} diversa da quella di 'label' {len(df['label'])}. Saltando questa colonna.")
            continue
        
        ce = conditional_entropy(df[col], df["label"])
        eccd = global_ent - ce
        print(f"Feature: {col}, ECCD = {eccd:.4f}")
        
        if eccd >= threshold:
            selected_features.append(col)
    
    print(f"🔹 Numero di feature selezionate: {len(selected_features)} su {len(feature_columns)}")
    
    # Crea il nuovo DataFrame ridotto mantenendo solo le feature selezionate insieme a ID e Label
    df_reduced = df[["id", "label"] + selected_features]
    
    # Salva il nuovo CSV
    df_reduced.to_csv(output_csv, index=False, sep=";")
    print(f"✅ Dataset ridotto salvato in {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Riduce la dimensionalità del dataset mediante ECCD.")
    parser.add_argument("input_csv", help="File CSV con i vettori numerici (ID, Label, B-mer frequenze).")
    parser.add_argument("-o", "--output_csv", required=True, help="Nome del file CSV di output ridotto.")
    parser.add_argument("--threshold", type=float, default=0.02, help="Soglia minima di ECCD per mantenere una feature (default: 0.02).")
    args = parser.parse_args()
    
    process_file(args.input_csv, args.output_csv, args.threshold)
