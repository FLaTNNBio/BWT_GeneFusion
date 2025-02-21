import random
import sys

def divide_in_blocchi(sequenza, lunghezza_blocco=2500):
    """Divide la sequenza in blocchi di lunghezza specificata"""
    return [sequenza[i:i+lunghezza_blocco] for i in range(0, len(sequenza), lunghezza_blocco)]

def scegli_blocchi_casuali(blocchi, num_blocchi=10):
    """Sceglie fino a num_blocchi distinti casuali"""
    return random.sample(blocchi, min(num_blocchi, len(blocchi)))

def scrivi_fasta(nome_file, sequenze):
    """Scrive le sequenze selezionate in un file FASTA"""
    with open(nome_file, "w") as f:
        for identificatore, blocchi in sequenze.items():
            for i, sequenza in enumerate(blocchi):
                f.write(f">{identificatore}_blocco{i+1}\n")  # Riga di intestazione FASTA
                for j in range(0, len(sequenza), 80):  # Formattazione a 80 caratteri per riga
                    f.write(sequenza[j:j+80] + "\n")

def leggi_fasta(nome_file):
    """Legge un file FASTA e restituisce un dizionario {ID: sequenza}"""
    sequenze = {}
    identificatore = None
    sequenza_corrente = []
    try:
        with open(nome_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if identificatore and sequenza_corrente:
                        sequenze[identificatore] = "".join(sequenza_corrente)
                    identificatore = line[1:]  # Rimuove il carattere '>'
                    sequenza_corrente = []
                else:
                    sequenza_corrente.append(line)
            if identificatore and sequenza_corrente:
                sequenze[identificatore] = "".join(sequenza_corrente)
        return sequenze
    except FileNotFoundError:
        print(f"Errore: il file '{nome_file}' non esiste.")
        sys.exit(1)
    except Exception as e:
        print(f"Errore durante la lettura del file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python dividiBlocchi.py <file_fasta>")
        sys.exit(1)
    
    fasta_file = sys.argv[1]
    sequenze = leggi_fasta(fasta_file)
    
    if not sequenze:
        print("Errore: il file FASTA è vuoto o non valido.")
        sys.exit(1)
    
    output_sequenze = {}
    for identificatore, sequenza in sequenze.items():
        blocchi = divide_in_blocchi(sequenza)
        blocchi_rappresentativi = scegli_blocchi_casuali(blocchi)
        if blocchi_rappresentativi:
            output_sequenze[identificatore] = blocchi_rappresentativi
    
    if output_sequenze:
        output_file = "output.fasta"
        scrivi_fasta(output_file, output_sequenze)
        print(f"File FASTA generato con blocchi selezionati: {output_file}")
    else:
        print("Errore: nessun blocco rappresentativo trovato!")

