import os
import json
import numpy as np
import tensorflow as tf

# ================= CONFIGURATION =================
TRAIN_DIR = r"dataset_split/train"
VAL_DIR   = r"dataset_split/val"

IMG_SIZE = (380, 380) # EfficientNetB4
BATCH = 128          
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

# -------------------------------------------------
# 0) Config GPU
# -------------------------------------------------
tf.keras.backend.clear_session()
try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

# -------------------------------------------------
# 1) Datasets
# -------------------------------------------------
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

# -------------------------------------------------
# 1.1) Sauvegarde des labels
# -------------------------------------------------
label2id = {name: idx for idx, name in enumerate(class_names)}
id2label = {idx: name for name, idx in label2id.items()}

with open("label2id.json", "w", encoding="utf-8") as f:
    json.dump(label2id, f, ensure_ascii=False, indent=2)

with open("id2label.json", "w", encoding="utf-8") as f:
    json.dump(id2label, f, ensure_ascii=False, indent=2)

print(" Fichiers générés : label2id.json et id2label.json")

# -------------------------------------------------
# 2) Class weights
# -------------------------------------------------
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

counts = np.zeros(num_classes, dtype=np.int64)
for i, cname in enumerate(class_names):
    folder = os.path.join(TRAIN_DIR, cname)
    if not os.path.isdir(folder):
        continue
    counts[i] = sum(1 for f in os.listdir(folder) if f.lower().endswith(IMG_EXT))

total = counts.sum()
class_weight = {
    i: float(total) / (num_classes * counts[i])
    for i in range(num_classes) if counts[i] > 0
}

print("Using class_weight:", class_weight)

# -------------------------------------------------
# 3) Pipeline
# -------------------------------------------------
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)

# -------------------------------------------------
# 4) Modèle
# -------------------------------------------------
data_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomContrast(0.10),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
], name="data_augmentation")

base = tf.keras.applications.EfficientNetB4(
    include_top=False,
    weights="imagenet",
    input_shape=IMG_SIZE + (3,)
)
base.trainable = False 

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_aug(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.30)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

# -------------------------------------------------
# 5) Callbacks & Logger
# -------------------------------------------------
# ICI: CSVLogger sauvegarde loss, accuracy, val_..., et lr
csv_logger = tf.keras.callbacks.CSVLogger("training_log.csv", separator=",", append=False)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("best.keras", save_best_only=True),
    tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=2, min_lr=1e-6),
    csv_logger # Ajout du logger
]

# -------------------------------------------------
# 6) Phase 1
# -------------------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n🚀 Phase 1 training (head)...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=7,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

# -------------------------------------------------
# 7) Phase 2
# -------------------------------------------------
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
for layer in base.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# Important : On met append=True pour ne pas effacer l'historique de la Phase 1
csv_logger_phase2 = tf.keras.callbacks.CSVLogger("training_log.csv", separator=",", append=True)

# Mise à jour de la liste des callbacks avec le nouveau logger
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("best.keras", save_best_only=True),
    tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=2, min_lr=1e-6),
    csv_logger_phase2
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nPhase 2 fine-tuning...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

model.save("plant_disease_classifier.keras")
print("Modèle sauvegardé. Historique sauvegardé dans 'training_log.csv'")