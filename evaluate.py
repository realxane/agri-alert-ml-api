import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

TEST_DIR   = r"dataset_split/test"
MODEL_PATH = "plant_disease_final_model.keras"

IMG_SIZE = (380, 380)
BATCH = 64 # Réduit à 16 pour éviter la saturation de la RAM (SEVERE STARVATION)

# 1) Charger le dataset test (IMPORTANT: shuffle=False)
print("Chargement des données de test...")
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH,
    shuffle=False
)

# 2) Charger le modèle
print("\nChargement du modèle...")
model = tf.keras.models.load_model(MODEL_PATH)

# 3) Évaluation globale (Accuracy)
print("\nCalcul de l'exactitude (Accuracy)...")
loss, acc = model.evaluate(test_ds, verbose=1)

# 4) Prédictions pour calculer le F1-score
print("\nGénération des prédictions pour le F1-score...")
test_eval_ds = test_ds.prefetch(tf.data.AUTOTUNE)
y_true = np.concatenate([y.numpy() for _, y in test_eval_ds], axis=0)
probs = model.predict(test_eval_ds, verbose=1)
y_pred = np.argmax(probs, axis=1)

# 5) Calcul du F1-score macro
macro_f1 = f1_score(y_true, y_pred, average="macro")

# 6) Affichage direct pour ton tableau
print("\n" + "="*60)
print(f"RÉSULTATS À COPIER DANS LE TABLEAU :")
print(f"Évaluation sur [{len(y_true)}] images de test, jamais vues pendant l'entraînement :")
print("-" * 60)
print(f"Exactitude (accuracy)      : {acc * 100:.2f} %")
print(f"F1-score (moyenne macro)   : {macro_f1 * 100:.2f} %")
print("="*60)