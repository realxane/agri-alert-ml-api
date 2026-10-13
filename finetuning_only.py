import os
import json
import numpy as np
import tensorflow as tf

TRAIN_DIR = r"C:\Users\souma\OneDrive\Documents\Alassane\IA_Alassane\dataset_split\train"
VAL_DIR   = r"C:\Users\souma\OneDrive\Documents\Alassane\IA_Alassane\dataset_split\val"

IMG_SIZE = (300, 300)
BATCH = 128
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

# (Optionnel) sécurité GPU
tf.keras.backend.clear_session()
try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

# -----------------------------
# 1) Datasets (mêmes paramètres que phase 1)
# -----------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Nombre de classes:", num_classes)

# -----------------------------
# 1.1) Recharger/valider mapping (important)
# -----------------------------
# Si label2id.json existe déjà, on le recharge (recommandé).
# Sinon, on le régénère.
if os.path.exists("label2id.json"):
    with open("label2id.json", "r", encoding="utf-8") as f:
        label2id = json.load(f)
    # Vérif rapide: mêmes classes
    if list(label2id.keys()) != class_names:
        print("⚠️ Attention: l'ordre des classes a changé !")
        print("JSON:", list(label2id.keys())[:5], "...")
        print("DS  :", class_names[:5], "...")
        print("➡️ Il faut garder EXACTEMENT le même ordre qu'à l'entraînement.")
else:
    label2id = {name: idx for idx, name in enumerate(class_names)}
    with open("label2id.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    print("✅ label2id.json régénéré")

# -----------------------------
# 2) Class weights (comme avant)
# -----------------------------
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
counts = np.zeros(num_classes, dtype=np.int64)

for i, cname in enumerate(class_names):
    folder = os.path.join(TRAIN_DIR, cname)
    if not os.path.isdir(folder):
        continue
    counts[i] = sum(1 for f in os.listdir(folder) if f.lower().endswith(IMG_EXT))

total = counts.sum()
class_weight = {i: float(total) / (num_classes * counts[i]) for i in range(num_classes) if counts[i] > 0}

# -----------------------------
# 3) Prefetch
# -----------------------------
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)

# -----------------------------
# 4) Charger le meilleur modèle de la phase 1
# -----------------------------
# best.keras a été sauvegardé pendant la phase 1 via ModelCheckpoint
model = tf.keras.models.load_model("best.keras")
print("✅ Modèle chargé: best.keras")

# -----------------------------
# 5) Retrouver la base EfficientNet dans le modèle
# -----------------------------
# Dans ton modèle, le backbone s'appelle souvent "efficientnetb3"
# On le retrouve de manière robuste en cherchant une couche qui contient "efficientnet"
base = None
for layer in model.layers:
    if "efficientnet" in layer.name.lower():
        base = layer
        break

if base is None:
    raise RuntimeError("Impossible de trouver la base EfficientNet dans le modèle chargé. "
                       "Affiche model.summary() et cherche le nom exact de la couche backbone.")

print("✅ Backbone trouvé:", base.name)

# -----------------------------
# 6) Défiger partiellement pour fine-tuning
# -----------------------------
base.trainable = True

# Défreeze seulement les dernières couches
for layer in base.layers[:-40]:
    layer.trainable = False

# Geler BatchNorm (recommandé)
for layer in base.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# -----------------------------
# 7) Recompile (LR petit)
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("best_finetuned.keras", save_best_only=True),
    tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=2, min_lr=1e-6),
]

# -----------------------------
# 8) Fine-tuning uniquement
# -----------------------------
print("\n🔥 Fine-tuning ONLY...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

model.save("plant_disease_classifier.keras")
print("✅ Modèle sauvegardé : plant_disease_classifier.keras")