import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools

# ================= CONFIGURATION =================
VAL_DIR = r"C:\Users\souma\OneDrive\Documents\Alassane\IA2_EfficientNetB4\dataset_split\val"
MODEL_PATH = "best.keras" 

IMG_SIZE = (380, 380)
BATCH = 128

# 1) Dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, image_size=IMG_SIZE, batch_size=BATCH, shuffle=False
)
class_names = val_ds.class_names
num_classes = len(class_names)

# 2) Charger le modèle
print(f"Chargement du modèle depuis {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

# 3) Prédictions
print("Lancement des prédictions...")
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
y_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0)
probs = model.predict(val_ds, verbose=1)
y_pred = np.argmax(probs, axis=1)

# 4) Matrice de confusion
cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes).numpy()

# Normalisation (pour les couleurs, mais on affichera les nombres bruts)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# Gérer la division par zéro si une classe est vide
cm_norm = np.nan_to_num(cm_norm)

# 5) Fonction d'affichage avec TEXTE
def plot_confusion_matrix(matrix, labels, title, filename, normalize=False):
    plt.figure(figsize=(20, 20)) # Taille augmentée pour lisibilité
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=16)
    plt.colorbar(fraction=0.046, pad=0.04)
    
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90, fontsize=8)
    plt.yticks(tick_marks, labels, fontsize=8)

    # ✅ AJOUT : Boucle pour écrire les nombres dans les cases
    fmt = '.2f' if normalize else 'd' # Format décimal ou entier
    thresh = matrix.max() / 2.
    
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        val = matrix[i, j]
        # On n'affiche le texte que si la valeur > 0 pour ne pas surcharger (optionnel, retirer le if pour tout afficher)
        if val > 0:
            plt.text(j, i, format(val, fmt),
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if val > thresh else "black",
                     fontsize=7) # Ajuster la taille de la police si nécessaire

    plt.ylabel('Vraie classe', fontsize=12)
    plt.xlabel('Classe prédite', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Image sauvegardée : {filename}")

# Affichage 1 : Nombres bruts (Counts)
plot_confusion_matrix(cm, class_names, "Matrice de Confusion (Nombres)", "confusion_matrix_counts.png", normalize=False)

# Affichage 2 : Normalisée (Pourcentages) - Optionnel si tu veux voir les % dans les cases
# plot_confusion_matrix(cm_norm, class_names, "Matrice de Confusion (Normalisée)", "confusion_matrix_norm.png", normalize=True)

# 6) Top erreurs (inchangé, toujours utile)
err = cm.copy()
np.fill_diagonal(err, 0)
flat_idx = np.argsort(err.ravel())[::-1]

print("\n--- Top 15 des confusions (Vrai -> Prédit : Nbr Erreurs) ---")
shown = 0
for idx in flat_idx:
    i, j = np.unravel_index(idx, err.shape)
    if err[i, j] == 0:
        break
    print(f"{class_names[i]} -> {class_names[j]} : {err[i, j]}")
    shown += 1
    if shown >= 15:
        break