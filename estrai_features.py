#!/usr/bin/env python
"""
Estrae i B-mers dalle sequenze BWT contenute in un file CSV.
"""

import argparse
import pandas as pd
from collections import defaultdict

def estrai_bmers(bwt_seq, min_len=2, max_len=6, min_freq=10):
    """
    Estrae i B-mers dalla sequenza BWT.
    - min_len: lunghezza minima dei B-mers
    - max_len: lunghezza massima dei B-mers
    - min_freq: frequenza minima (k) per cui un B-mer viene scelto
    """
    bmers = defaultdict(int)
    
    # Scorriamo la sequenza e consideriamo tutte le sottosequenze di lunghezza tra min_len e max_len
    for i in range(len(bwt_seq)):
        for k in range(min_len, max_len + 1):
            if i + k <= len(bwt_seq):
                mer = bwt_seq[i:i + k]
                bmers[mer] += 1

    # Filtriamo i B-mers che compaiono almeno min_freq volte
    bmers_filtrati = {mer: freq for mer, freq in bmers.items() if freq >= min_freq}
    
    return bmers_filtrati

def processa_bwt_file(input_csv, output_csv, min_len=2, max_len=6, min_freq=10):
    """
    Legge il file CSV con le sequenze BWT, estrae i B-mers e salva i risultati in un nuovo file CSV.
    """
    # Legge il CSV specificando il separatore corretto e la codifica
    try:
        df = pd.read_csv(input_csv, sep=";", encoding="utf-8")
    except UnicodeDecodeError:
        print("⚠️ Problema con UTF-8, provo con ISO-8859-1...")
        df = pd.read_csv(input_csv, sep=";", encoding="ISO-8859-1")
    
    # Stampa i nomi delle colonne per verificare eventuali errori
    print("🔍 Colonne trovate nel CSV:", df.columns.tolist())
    
    # Rimuove eventuali spazi dai nomi delle colonne
    df.columns = df.columns.str.strip()
    
    if "BWT" not in df.columns:
        raise KeyError("❌ ERRORE: La colonna 'BWT' non è presente nel file CSV. Controlla il nome delle colonne.")
    
    results = []
    
    for _, row in df.iterrows():
        bwt_seq = row["BWT"]
        bmers = estrai_bmers(bwt_seq, min_len, max_len, min_freq)
        for mer, freq in bmers.items():
            results.append([row["ID"], mer, freq, row["Label"]])
    
    df_bmers = pd.DataFrame(results, columns=["ID", "B-mer", "Frequency", "Label"])
    df_bmers.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"✅ B-mers estratti e salvati in {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estrae i B-mers dalle sequenze BWT in un file CSV.")
    parser.add_argument("input_csv", help="File CSV con le sequenze BWT.")
    parser.add_argument("-o", "--output_csv", required=True, help="Nome del file CSV con i B-mers estratti.")
    parser.add_argument("--min_len", type=int, default=2, help="Lunghezza minima dei B-mers (default: 2).")
    parser.add_argument("--max_len", type=int, default=6, help="Lunghezza massima dei B-mers (default: 6).")
    parser.add_argument("--min_freq", type=int, default=10, help="Frequenza minima (k) per cui un B-mer viene scelto (default: 10).")
    
    args = parser.parse_args()
    
    processa_bwt_file(args.input_csv, args.output_csv, args.min_len, args.max_len, args.min_freq)
