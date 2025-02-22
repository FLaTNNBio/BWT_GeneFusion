# 📌 Progetto: BWT_GeneFusion

## ⚙️ Requisiti

### 📌 Versione Python
Il progetto è stato sviluppato e testato con **Python 3.10+**.

### 📦 Pacchetti richiesti
Per installare tutti i pacchetti necessari, eseguire:
```bash
pip install scikit-learn pandas numpy seaborn matplotlib biopython
```

- **scikit-learn** → Per l'addestramento dei modelli di Machine Learning (SVM, Naive Bayes, Rete Neurale)
- **pandas** → Per la gestione dei dataset CSV
- **numpy** → Per operazioni matematiche avanzate
- **seaborn & matplotlib** → Per la visualizzazione dei risultati (grafici e matrici di confusione)
- **biopython** → Per la gestione dei file FASTA

---
### 📌 File di partenza 
fusim_bench.fasta : contiene le sequenze che presentano gene fusion
all_transcripts.fasta : contiene le sequenze che non presentano gene fusion 
genes_panel.txt : elenco di geni delle sequenze
---

## 📂 Workflow e Utilizzo degli Script

### 1️⃣ **Applicazione della BWT alle sequenze**
**Script:** `bwt_input_fasta.py`

**Utilizzo:**
```bash
python bwt_input_fasta.py fusim_bench.fasta -o fusim_bench_bwt.csv
```
📌 **Output:**
- `fusim_bench_bwt.csv` → Sequenze con gene fusion con BWT applicata

---

### 2️⃣ **Divisione delle sequenze lunghe (senza gene fusion) in blocchi**
**Script:** `dividiInBlocchi.py`

**Utilizzo:**
```bash
python dividiInBlocchi.py all_transcripts.fasta -o all_transcripts_blocks.fasta
```
📌 **Output:**
- `all_transcripts_blocks.fasta` → Sequenze lunghe divise in blocchi più corti per uniformarle a quelle con gene fusion
                                   il numero di blocchi scelto per ogni sequenza è stato scelto per ottenere un numero simile agli elementi 
                                   di fusim_bench.fasta
---

### 3️⃣ **Applicazione della BWT ai blocchi generati**
**Script:** `bwt_input_fasta.py`

**Utilizzo:**
```bash
python bwt_input_fasta.py all_transcripts_blocks.fasta -o all_transcripts_blocks_bwt.csv
```
📌 **Output:**
- `all_transcripts_blocks_bwt.csv` → BWT applicata ai blocchi

📌 **Nota:** I due CSV (`fusim_bench_bwt.csv` e `all_transcripts_blocks_bwt.csv`) vengono poi uniti manualmente in un unico file full_dataset.csv.
    Nel file full_dataset.csv abbiamo anche una colonna "Label" che indica la tipologia di ogni sequenza (0 = no gene fusion, 1 = gene fusion)

---

### 4️⃣ **Estrazione dei B-mers dalle sequenze BWT**
**Script:** `estrai_features.py`

**Utilizzo:**
```bash
python estrai_features.py full_dataset.csv -o full_dataset_bmers.csv --min_freq k (specifica valore k)
```
📌 **Output:**
- `full_dataset_bmers.csv` → File con i B-mers estratti, che compaiono almeno **k volte** in una sequenza

---

### 5️⃣ **Creazione dei vettori numerici per il Machine Learning**
**Script:** `crea_vettoriNumerici.py`

**Utilizzo:**
```bash
python crea_vettoriNumerici.py full_dataset_bmers.csv -o full_dataset_vectors.csv
```
📌 **Output:**
- `full_dataset_vectors.csv` → Dataset che organizza le sequenze sulle righe ed i b-mers sulle colonne, 
                               ogni cella contiene il numero di volte che un bmer j-esimo compare in una sequenza i-esima

---

### 6️⃣ **Arrotondamento dei valori con numeri periodici**
**Script:** `arrotondamento_periodici.py`

**Utilizzo:**
```bash
python arrotondamento_periodici.py full_dataset_vectors.csv -o full_dataset_vectors_rounded.csv
```
📌 **Output:**
- `full_dataset_vectors_rounded.csv` → Dataset con valori arrotondati

---

### 7️⃣ **Riduzione del dataset con ECCD (Entropy-based Category Coverage Difference)**
**Script:** `eccd.py`

**Utilizzo:**
```bash
python eccd.py full_dataset_vectors_rounded.csv -o full_dataset_vectors_reduced.csv
```
📌 **Output:**
- `full_dataset_vectors_reduced.csv` → Dataset ridotto eliminando le feature meno informative

---

### 8️⃣ **Addestramento dei modelli di Machine Learning**

#### 🔹 **Support Vector Machine (SVM)**
**Script:** `train_svm.py`

**Utilizzo:**
```bash
python train_svm.py full_dataset_vectors_reduced.csv
```
📌 **Output:**
- Stampa **accuracy**, **precision**, **recall** e **f1-score**
- Genera **matrice di confusione** come immagine PNG

#### 🔹 **Naive Bayes**
**Script:** `train_naiveBayes.py`

**Utilizzo:**
```bash
python train_naiveBayes.py full_dataset_vectors_reduced.csv
```
📌 **Output:**
- Stampa **accuracy**, **precision**, **recall** e **f1-score**
- Genera **matrice di confusione** come immagine PNG

#### 🔹 **Rete Neurale (MLPClassifier)**
**Script:** `train_reteNeurale.py`

**Utilizzo:**
```bash
python train_reteNeurale.py full_dataset_vectors_reduced.csv
```
📌 **Output:**
- Stampa **accuracy**, **precision**, **recall** e **f1-score**
- Genera **matrice di confusione** come immagine PNG


#### 🔹 **Random Forest**
**Script:** `train_randomForest.py`

**Utilizzo:**
```bash
python train_randomForest.py full_dataset_vectors_reduced.csv
```
📌 **Output:**
- Stampa **accuracy**, **precision**, **recall** e **f1-score**
- Genera **matrice di confusione** come immagine PNG



---

## 📊 Confronto dei Modelli
Dopo aver eseguito gli script di addestramento, è possibile confrontare i risultati ottenuti dai diversi algoritmi


---

## 📌 Conclusione
Questo progetto consente di estrarre dalla BWT delle sequenze a disposizione i b-mers che vengono usati come dati per l'addestramento e il test di diversi algoritmi di machine learning.
Le prestazioni, riguardanti la classificazione delle sequenze in due classi (0: no gene fusion, 1 gene fusion), degli algoritmi, vengono confrontate per visualizzare quale modello classifica con maggiore accuratezza.

📩 Contatti: v.tarantino7@studenti.unisa.it ,  bmazzeo3@studenti.unisa.it
