import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, f1_score

TEST_DIR   = r"C:\Users\souma\OneDrive\Documents\Alassane\IA2_EfficientNetB4\dataset_split\test"
MODEL_PATH = "best.keras"  # ou "best.keras"

IMG_SIZE = (380, 380)
BATCH = 128

# 1) Charger le dataset test (IMPORTANT: shuffle=False)
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH,
    shuffle=False
)
class_names = test_ds.class_names

# 2) Charger le modèle
model = tf.keras.models.load_model(MODEL_PATH)

# 3) Évaluation globale (loss + accuracy)
loss, acc = model.evaluate(test_ds, verbose=1)
print(f"\n✅ Test accuracy = {acc:.4f} | test loss = {loss:.4f}")

# 4) Prédictions pour métriques détaillées
test_eval_ds = test_ds.prefetch(tf.data.AUTOTUNE)

y_true = np.concatenate([y.numpy() for _, y in test_eval_ds], axis=0)
probs = model.predict(test_eval_ds, verbose=1)
y_pred = np.argmax(probs, axis=1)

# 5) Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion matrix shape:", cm.shape)  # (nb_classes, nb_classes)

# 6) Precision / Recall / F1 par classe
print("\n=== Classification report (par classe) ===")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# 7) Scores globaux
macro_f1 = f1_score(y_true, y_pred, average="macro")
weighted_f1 = f1_score(y_true, y_pred, average="weighted")
print(f"Macro F1   : {macro_f1:.4f}  (chaque classe compte pareil)")
print(f"Weighted F1: {weighted_f1:.4f} (pondéré par le nb d’images/classe)")