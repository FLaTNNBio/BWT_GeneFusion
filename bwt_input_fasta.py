#!/usr/bin/env python
"""
Burrows-Wheeler Transform (BWT) applicata a file FASTA.
Esegue la BWT su ogni sequenza del file FASTA e salva i risultati in un file CSV.

Usage:
    python bwt.py <fasta_file> [-o <output_file>]
"""

import argparse
from Bio import SeqIO
from operator import itemgetter
import pandas as pd

def bw_transform(s):
    """Esegue la Burrows-Wheeler Transform su una stringa s."""
    n = len(s)
    m = sorted([s[i:] + s[:i] for i in range(n)])  # Costruisce la matrice delle rotazioni
    I = m.index(s)  # Indice della stringa originale nella matrice ordinata
    L = ''.join([q[-1] for q in m])  # Ultima colonna della matrice
    return I, L

def bw_restore(I, L):
    """Ricostruisce la stringa originale dalla BWT."""
    n = len(L)
    X = sorted([(i, x) for i, x in enumerate(L)], key=itemgetter(1))
    
    T = [None for _ in range(n)]
    for i, y in enumerate(X):
        j, _ = y
        T[j] = i

    Tx = [I]
    for i in range(1, n):
        Tx.append(T[Tx[i-1]])
        
    S = [L[i] for i in Tx]
    S.reverse()
    return ''.join(S)

def process_fasta_bwt(fasta_file):
    """
    Legge il file FASTA e per ogni record:
      - Estrae la sequenza originale
      - Applica la BWT per ottenere indice e sequenza trasformata
      - Stampa i risultati (ID, Indice, BWT)
    """
    results = []  # Lista per salvare i risultati
    for record in SeqIO.parse(fasta_file, "fasta"):
        original_seq = str(record.seq)
        index, bwt_seq = bw_transform(original_seq)
        results.append((record.id, index, bwt_seq))
        print(f"ID: {record.id} | Indice: {index} | BWT: {bwt_seq}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Applica la BWT ad un file FASTA e salva i risultati in un CSV")
    parser.add_argument("fasta_file", help="Nome o percorso del file FASTA")
    parser.add_argument("-o", "--output", help="Nome del file CSV di output (opzionale)")
    args = parser.parse_args()
    
    results = process_fasta_bwt(args.fasta_file)
    
    df = pd.DataFrame(results, columns=["ID", "Indice", "BWT"])
    # Se non viene specificato un file di output, creiamo un nome a partire dal file FASTA
    output_file = args.output if args.output else args.fasta_file.replace(".fasta", "_BWT.csv")
    df.to_csv(output_file, index=False)
    print(f"Risultati salvati in {output_file}")

if __name__ == "__main__":
    main()

