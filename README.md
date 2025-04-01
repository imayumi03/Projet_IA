# 🤖 Prédiction des Arrêts de Protection d’un Cobot UR3

## 👤 Réalisé par

- **Nom :** [Ton Nom]
- **Prénom :** [Ton Prénom]

## 🧠 Objectif du projet

Développer une solution d'intelligence artificielle pour prédire les arrêts de protection du cobot UR3 à partir des données capteurs temporelles. Le modèle doit anticiper un arrêt à partir des 10 dernières unités de temps.

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
- Détection et traitement des outliers (méthodes IQR, Z-score)
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
  - SARIMA (pour séries univariées)
  - Random Forest
  - XGBoost / LightGBM
- Évaluation par :
  - Accuracy, precision, recall, F1-score
  - ROC-AUC
  - Courbes ROC, matrice de confusion

### 🌐 6. Création d’une API Flask

- API REST avec route `/predict`
- Entrée : JSON contenant une séquence temporelle
- Sortie : Probabilité d’un arrêt de protection
- Exemple :
  ```bash
  curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"sequence": [...]}'
