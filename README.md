# Projet_IA
Ce projet a pour objectif de prédire les arrêts de protection d’un robot collaboratif UR3 à partir de données temporelles multivariées issues de capteurs.

# 🔧 Méthodologie adoptée
1. Exploration et prétraitement des données
Chargement des données UR3 depuis le jeu de données UR3 CobotOps.
Nettoyage : suppression des lignes contenant des valeurs manquantes.
Conversion de la colonne Timestamp en datetime avec vérification de l'ordre temporel.
Normalisation des variables numériques via StandardScaler.

2. Équilibrage des classes avec SMOTE
Le dataset initial était fortement déséquilibré (classe 1 = ~4%).
Utilisation de la méthode SMOTE pour sur-échantillonner la classe minoritaire avant la création des séquences temporelles.

3. Construction des séquences temporelles
Utilisation d’une fenêtre glissante de taille 10 pour transformer les données tabulaires en séquences compatibles avec les RNN (LSTM, GRU).

4. Division du dataset
Séparation en 80% train / 20% test avec stratify=y pour maintenir la répartition des classes.

🧳️ Modèles et hyperparamètres testés

1. LSTM
hidden_size = 64, num_layers = 2, dropout = 0.3, batch_size = 64, lr = 0.001
Perte : BCEWithLogitsLoss()
Entraînement sur 20 époques

2. Comparaison avec LSTM + pos_weight (sans SMOTE)
Moins performant : recall faible pour la classe 1
Meilleure précision mais détection partielle des arrêts

3. Performance finale (avec SMOTE)
Accuracy : 98.62%
F1-score classe 1 : 98.62%
AUC ROC : 0.9987

# 🚀 API Flask & Instructions

1. Structure du projet

project/
|-- app.py
|-- model.pt
|-- scaler.pkl
|-- requirements.txt
|-- README.md

2. Lancer l’API Flask

pip install -r requirements.txt
python app.py

3. Utilisation de l’API

Endpoint : POST /predict

Input : JSON contenant un tableau features de taille (10, nb_features)

Exemple :

{
  "features": [[...], [...], ..., [...]]
}

Output : probabilité de prédiction + classe prédite

4. Enregistrement / chargement du modèle

# Sauvegarde
torch.save(model.state_dict(), 'model.pt')

# Chargement
model.load_state_dict(torch.load('model.pt'))
model.eval()

📊 Auteurs & Licence

Projet réalisé dans le cadre du module Intelligence Artificielle et Industrie 4.0.

Licence : MIT
