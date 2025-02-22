import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse

# 1. Caricare il dataset
def load_dataset(csv_file):
    df = pd.read_csv(csv_file, sep=";", encoding="utf-8")
    print(f"\U0001F4A1 Dataset caricato: {df.shape[0]} righe, {df.shape[1]} colonne")
    
    X = df.drop(columns=["id", "label"])
    y = df["label"]
    
    return X, y

# 2. Dividere il dataset in training e test
def split_data(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"\U0001F4A1 Suddivisione dati: {X_train.shape[0]} train, {X_test.shape[0]} test")
    return X_train, X_test, y_train, y_test

# 3. Normalizzare i dati
def normalize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# 4. Addestrare la Random Forest
def train_random_forest(X_train, y_train, n_estimators=100):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    return model

# 5. Valutare il modello e visualizzare i risultati
def evaluate_model(model, X_test, y_test, feature_names):
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}\n")
    
    print("\U0001F4A1 Report di classificazione:")
    print(classification_report(y_test, y_pred))
    
    print("\U0001F4A1 Matrice di confusione:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Visualizzazione Matrice di Confusione
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel("Predetto")
    plt.ylabel("Reale")
    plt.title("Matrice di Confusione")
    plt.show()
    
    # Visualizzazione Importanza delle Feature
    feature_importances = model.feature_importances_
    sorted_idx = np.argsort(feature_importances)[::-1]
    plt.figure(figsize=(10, 5))
    sns.barplot(x=feature_importances[sorted_idx][:10], y=np.array(feature_names)[sorted_idx][:10], palette='viridis')
    plt.xlabel("Importanza")
    plt.ylabel("Feature")
    plt.title("Top 10 Feature Importanti")
    plt.show()

# 6. Funzione principale
def main(input_csv):
    X, y = load_dataset(input_csv)
    X.fillna(0, inplace=True)
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)
    
    model = train_random_forest(X_train_scaled, y_train)
    evaluate_model(model, X_test_scaled, y_test, X.columns)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Addestra una Random Forest sul dataset di sequenze genetiche filtrato dopo ECCD."
    )
    parser.add_argument("input_csv", help="File CSV con features selezionate e label (dopo ECCD).")
    args = parser.parse_args()

    main(args.input_csv)
