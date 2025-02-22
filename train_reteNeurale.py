import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse

# 1. Caricare il dataset
def load_dataset(csv_file):
    df = pd.read_csv(csv_file, sep=";", encoding="utf-8")
    print(f"🔹 Dataset caricato: {df.shape[0]} righe, {df.shape[1]} colonne")
    
    X = df.drop(columns=["id", "label"])  # Features (B-mers)
    y = df["label"]  # Target (0 = no fusion, 1 = fusion)
    
    return X, y

# 2. Dividere il dataset in training e test
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

# 4. Addestrare la Rete Neurale (MLP)
def train_mlp(X_train, y_train):
    model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', 
                          max_iter=500, random_state=42) #Un livello nascosto con 100 neuroni.
                                                         #Funzione di attivazione: ReLU (più veloce e stabile rispetto a sigmoid/tanh).
                                                         #Ottimizzatore: Adam, una variante avanzata della discesa del gradiente.
                                                         #500 epoche: L’addestramento dura 500 iterazioni 
    model.fit(X_train, y_train)
    return model

# 5. Creare il grafico della matrice di confusione
def crea_grafico(cm, acc, output_image="confusion_matrix_mlp.png"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Fusion (0)", "Fusion (1)"],
                yticklabels=["No Fusion (0)", "Fusion (1)"])
    plt.xlabel("Classe Predetta")
    plt.ylabel("Classe Reale")
    plt.title(f"Matrice di Confusione MLP (Accuracy: {acc:.4f})")
    plt.savefig(output_image, dpi=300)
    print(f"📊 Grafico salvato come '{output_image}'")
    plt.show()

# 6. Valutare il modello
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}\n")

    print("🔹 Report di classificazione:")
    print(classification_report(y_test, y_pred))

    print("🔹 Matrice di confusione:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    crea_grafico(cm, acc)

# 7. Funzione principale
def main(input_csv):
    X, y = load_dataset(input_csv)
    
    X.fillna(0, inplace=True)  # Sostituisce eventuali NaN con 0
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)
    
    model = train_mlp(X_train_scaled, y_train)
    
    evaluate_model(model, X_test_scaled, y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Addestra una rete neurale MLP sul dataset di sequenze genetiche filtrato dopo ECCD."
    )
    parser.add_argument("input_csv", help="File CSV con features selezionate e label (dopo ECCD).")
    args = parser.parse_args()
    
    main(args.input_csv)
