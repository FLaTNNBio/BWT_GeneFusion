import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Caricare il dataset
def load_dataset(csv_file):
    df = pd.read_csv(csv_file, sep=";", encoding="utf-8")
    print(f"🔹 Dataset caricato: {df.shape[0]} righe, {df.shape[1]} colonne")
    
    X = df.drop(columns=["id", "label"])
    y = df["label"]
    
    return X, y

# 2. Suddividere il dataset in training e test
def split_data(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"🔹 Suddivisione dati: {X_train.shape[0]} train, {X_test.shape[0]} test")
    return X_train, X_test, y_train, y_test

# 3. Normalizzare i dati (Naive Bayes non richiede normalizzazione, ma trasformiamo i NaN)
def preprocess_data(X_train, X_test):
    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)
    return X_train, X_test

# 4. Addestrare il modello Naive Bayes
def train_naive_bayes(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

# 5. Valutare il modello
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}\n")
    
    print("🔹 Report di classificazione:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    print("🔹 Matrice di confusione:")
    print(cm)

    # Creazione del grafico della matrice di confusione
    crea_grafico(cm, acc)

# 6. Creare grafico della matrice di confusione
def crea_grafico(cm, acc):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Fusion", "Fusion"], 
                yticklabels=["No Fusion", "Fusion"])
    plt.xlabel("Predetto")
    plt.ylabel("Reale")
    plt.title(f"Naive Bayes - Matrice di Confusione (Accuracy: {acc:.4f})")
    plt.savefig("naive_bayes_confusion_matrix.png")
    print("📊 Grafico salvato come 'naive_bayes_confusion_matrix.png'")
    plt.show()

# 7. Funzione principale
def main(input_csv):
    # Carica il dataset
    X, y = load_dataset(input_csv)

    # Suddivide in training e test
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Pre-elaborazione dati (sostituisce NaN con 0)
    X_train, X_test = preprocess_data(X_train, X_test)

    # Addestra Naive Bayes
    model = train_naive_bayes(X_train, y_train)

    # Valuta il modello
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Addestra un Naive Bayes sul dataset di sequenze genetiche filtrato dopo ECCD."
    )
    parser.add_argument("input_csv", help="File CSV con features selezionate e label (dopo ECCD).")
    args = parser.parse_args()

    main(args.input_csv)
