# Agri Alert — Diagnostic des maladies des plantes par Deep Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange) ![Flask](https://img.shields.io/badge/Flask-API-black)

## Description

Agri Alert est un outil d'aide à la décision pour les agriculteurs : à partir d'une photo de feuille, il identifie la maladie de la plante et propose un traitement adapté.

Ce dépôt contient la partie Machine Learning : l'entraînement du modèle et l'API de prédiction. Le projet complet comprend aussi :
- [Backend_AgriAlert](https://github.com/realxane/Backend_AgriAlert) : API Laravel (utilisateurs, historique) ;
- [Frontend_AgriAlert](https://github.com/realxane/Frontend_AgriAlert) : application mobile Flutter.

## Modèle

- **Architecture** : EfficientNetB4 pré-entraîné sur ImageNet, avec apprentissage par transfert (fine-tuning).
- **Données** : [nom du jeu de données, source], [N] images réparties en 37 classes (maladies et plantes saines).
- **Découpage** : [X] % entraînement, [Y] % validation, [Z] % test.
- **Augmentation de données** : [rotations, retournements, zoom…].

## Résultats

Évaluation sur [N] images de test, jamais vues pendant l'entraînement :

| Métrique | Valeur |
| --- | --- |
| Exactitude (accuracy) | [..] % |
| F1-score (moyenne macro) | [..] % |

![Courbes d'entraînement](training_curves.png)
![Matrice de confusion](confusion_matrix_counts.png)

## Architecture de l'API

1. **Entraînement** (`[train.ipynb]`) : préparation des données, fine-tuning d'EfficientNetB4, évaluation.
2. **API de prédiction** (`app.py`) : API REST Flask qui reçoit une image et renvoie la maladie détectée avec son score de confiance.
3. **Recommandations** (`recommendations.json`) : association de chaque maladie à un traitement conseillé.

## Lancement local

Le modèle entraîné (`plant_disease_final_model.keras`) est disponible sous le nom de ('best.keras') [https://drive.google.com/drive/folders/1_p-oxv8p93n1NRHo3g71iSH0cDeZ2AqE?usp=drive_link](lien-drive). Placez-le à la racine du projet, puis :

```bash
pip install -r requirements.txt
python app.py
```

## Limites et pistes d'amélioration

- Les images d'entraînement sont prises [en conditions contrôlées / sur fond uni] : les performances sur des photos de terrain au Mali restent à valider.
- Pistes : enrichir le jeu de données avec des cultures locales (mil, sorgho, coton), et alléger le modèle pour une utilisation hors ligne sur téléphone.