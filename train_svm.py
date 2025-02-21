import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
import matplotlib.pyplot as plt


# 1. Caricare il dataset
def load_dataset(csv_file):
    # Legge il file CSV usando ";" come separatore e codifica UTF-8
    df = pd.read_csv(csv_file, sep=";", encoding="utf-8")
    print(f"🔹 Dataset caricato: {df.shape[0]} righe, {df.shape[1]} colonne")
    
    # Separiamo le feature dalla label
    X = df.drop(columns=["id", "label"])  #rimuove le colonne id e labellasciando solo i bmers
    y = df["label"] #la colonna label che contiene le classi (0 = no fusion, 1 = fusion)
    #label è utile per identificare le classi ma si porebbe fare anche vedendo l'inizio dell'id 
   
    return X, y

# 2. Dividere il dataset in training e test
#divido X (bmers) in train e test e lo stesso con le label
#faccio una stratificazione ovvero mantengo la stessa proporzione del ds originale (100 -> 80-20, 80->48-32)
def split_data(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"🔹 Suddivisione dati: {X_train.shape[0]} train, {X_test.shape[0]} test")
    return X_train, X_test, y_train, y_test

# 3. Normalizzare i dati
def normalize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# 4. Addestrare la SVM
#passi i bmers e le label di addestramento al modello basato su SVM
def train_svm(X_train, y_train, kernel="linear"):
    model = SVC(kernel=kernel, C=1.0, random_state=42)  # SVM con kernel lineare, usiamo kernel lineare perchè
    #i dati devono essere separati da una linea retta
    model.fit(X_train, y_train)
    return model
    
    
# Creare il grafico della matrice di confusione
def crea_grafico(cm, acc, output_image="confusion_matrix.png"):
    """
    Crea una heatmap della matrice di confusione e la salva come immagine PNG.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Fusion (0)", "Fusion (1)"],
                yticklabels=["No Fusion (0)", "Fusion (1)"])
    plt.xlabel("Classe Predetta")
    plt.ylabel("Classe Reale")
    plt.title(f"Matrice di Confusione (Accuracy: {acc:.4f})")
    plt.savefig(output_image, dpi=300)
    print(f"📊 Grafico salvato come '{output_image}'")
    plt.show()


# 5. Valutare il modello dopo l'addestramento 
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test) #si usa il modello sui bmers di test che il modello non 
                                   #ha mai visto prima, y_pred conterrà i risultati previsti dal modello,
                                   #ovvero l’etichetta prevista per ogni sequenza (0 = senza gene fusion, 1 = con gene fusion)
    acc = accuracy_score(y_test, y_pred) #misura il livello di accuratezza della classificazione dei dati di test
                                          #confrontando label predette con quelle reali
    print(f"✅ Accuracy: {acc:.4f}\n")
    
    print("🔹 Report di classificazione:")
    print(classification_report(y_test, y_pred))       #Precision: quante delle sequenze classificate come "gene fusion/no gene fusion" sono realmente corrette
                                                       #Recall: quante delle sequenze che realmente hanno "gene fusion/no gene fusion" sono state trovate dal modello
                                                       #F1-score: media tra precision e recall.
                                                       #Support: numero di esempi reali per ogni classe (quante 0 e quante 1 nel test set).
    
    print("🔹 Matrice di confusione:")                 #matrice che ha due righe (reali 0 e reali 1) e due colonne predetto 0 e predetto 1 
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    crea_grafico(cm, acc)





# 6. Funzione principale
def main(input_csv):
    # Carica il dataset
    X, y = load_dataset(input_csv)
    
    # Gestione dei valori NaN: sostituisce i NaN con 0
    X.fillna(0, inplace=True)
    
    # Dividi in training e test
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Normalizza i dati
    X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)
    
    # Addestra la SVM
    model = train_svm(X_train_scaled, y_train)
    
    #print(y.count)
    #print(y_test.count)
    
    # Valuta il modello
    evaluate_model(model, X_test_scaled, y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Addestra una SVM sul dataset di sequenze genetiche filtrato dopo ECCD."
    )
    parser.add_argument("input_csv", help="File CSV con features selezionate e label (dopo ECCD).")
    args = parser.parse_args()
    main(args.input_csv)
