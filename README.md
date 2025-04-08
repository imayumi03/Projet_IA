# 🤖 Prédiction des Arrêts de Protection d’un Cobot UR3

## 👤 Réalisé par

- **Nom :** Abdelmoumni
- **Prénom :** Mounia

## 🧠 Objectif du projet

Développer un modèle de ML\DL pour prédire les arrêts de protection du cobot UR3 à partir des données capteurs temporelles. Le modèle doit anticiper un arrêt à partir des 10 dernières unités de temps.

---

## 🧪 Méthodologie Adoptée

### 📊 1. Analyse exploratoire et statistique des données

- Chargement des données depuis [UCI Repository](https://archive.ics.uci.edu/dataset/963/ur3+cobotops)
- Analyse descriptive :
  - Moyenne, médiane, écart-type, min/max
  - Corrélations via heatmap
- Visualisations :
  - Histogrammes
  - Boxplots pour détection des outliers
  - Séries temporelles
  - PCA pour visualisation des clusters

### 🧹 2. Prétraitement des données

- Traitement des valeurs manquantes (imputation ou suppression)
- Détection des outliers
- Normalisation des données (MinMaxScaler, StandardScaler)
- Génération de séquences temporelles de taille 10
- Création de labels binaires pour classification
- Séparation en `train`, `validation` et `test`

### 🤖 3. Modélisation avec LSTM

- Architecture du modèle :
  - Couches LSTM ou Bidirectional LSTM
  - Dropout pour régularisation
  - Dense avec activation `sigmoid`
- Compilation :
  - Perte : `binary_crossentropy`
  - Optimiseur : `Adam`
  - Métriques : `accuracy`, `recall`, `precision`, `AUC`

### ⚙️ 4. Optimisation des hyperparamètres

- Recherche par `Grid Search` ou `Random Search`
- Paramètres testés :
  - Nombre de neurones LSTM
  - Taille de batch
  - Taux de dropout
  - Nombre d’époques
  - Learning rate
- Visualisation avec `Matplotlib` des courbes d'entraînement/validation

### 🔄 5. Comparaison avec d'autres modèles

- Modèles alternatifs :
  - Random Forest
  - XGBoost / LightGBM
- Évaluation par :
  - Accuracy, precision, recall, F1-score
  - ROC-AUC
  - Courbes ROC

## 🌐 Création d’une API Flask  
Pré-requis :  
- Python ≥ 3.7  
- Bibliothèques : flask, torch, numpy  
- Fichiers nécessaires :` app.py`, `model.py`, `lstm_trained.pth`  

1- Lancer l’API : ` python app.p`  
Cela démarre un serveur local sur : http://127.0.0.1:5000  

2- Faire une requête POST à /predict :  
Le JSON envoyé doit contenir une clé "sequence"  
avec une liste de vecteurs de dimension (window_size, input_size) ( 10 lignes × 18 colonnes)  
` {
  "sequence": [
    [0.1, 0.2, ..., 0.9],
    [0.3, 0.4, ..., 0.8],
    ...
  ]
} `

3- Tester avec curl :  
` curl -X POST http://127.0.0.1:5000/predict \  
  -H "Content-Type: application/json" \  
  -d @test_input.json `

4- Erreurs courantes à éviter :  
`"KeyError: 'sequence'"` → la clé "sequence" est manquante dans le JSON  
`shape mismatch` → la séquence ne respecte pas la forme (10, 18)  
