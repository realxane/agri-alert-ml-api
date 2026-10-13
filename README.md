# 🌿 Agri Alert - API Machine Learning

API Flask servant le modèle de Deep Learning pour le projet Agri Alert, une solution d'aide à la décision agricole.

## ⚙️ Fonctionnalités
- Chargement d'un modèle **EfficientNetB4** (`plant_disease_final_model.keras`) entraîné sur 37 classes de maladies de plantes.
- API REST avec Flask pour recevoir des images et retourner la maladie détectée.
- Reconstruction des classes via `class_indices.json` et des recommandations via `recommendations.json`.

## 🚀 Lancement local
```bash
pip install -r requirements.txt
python app.py