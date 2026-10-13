import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Charger les données d'entraînement
log_file = "training_log.csv"

if not os.path.exists(log_file):
    print(f"❌ Erreur : Le fichier {log_file} est introuvable.")
    print("Assure-toi d'avoir lancé l'entraînement avec le code précédent qui contenait 'CSVLogger'.")
    exit()

data = pd.read_csv(log_file)

# 2. Configurer le style des graphiques
plt.style.use('seaborn-v0_8-whitegrid') # Un style propre avec grille (similaire à ton image)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- Graphique 1 : Précision (Accuracy) ---
ax1.plot(data['accuracy'], label='Entraînement', color='#2ca02c', linewidth=2) # Vert
ax1.plot(data['val_accuracy'], label='Validation', color='#ff7f0e', linewidth=2) # Orange
ax1.set_title('Précision du Modèle (EfficientNetB4)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Époques', fontsize=12)
ax1.set_ylabel('Score', fontsize=12)
ax1.legend(loc='lower right', frameon=True)
ax1.grid(True, linestyle='--', alpha=0.6)

# --- Graphique 2 : Perte (Loss) ---
ax2.plot(data['loss'], label='Entraînement', color='#d62728', linewidth=2) # Rouge
ax2.plot(data['val_loss'], label='Validation', color='#1f77b4', linewidth=2) # Bleu
ax2.set_title('Perte du Modèle (Loss)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Époques', fontsize=12)
ax2.set_ylabel('Valeur de Perte', fontsize=12)
ax2.legend(loc='upper right', frameon=True)
ax2.grid(True, linestyle='--', alpha=0.6)

# 3. Sauvegarder et Afficher
plt.tight_layout()
save_path = "training_curves.png"
plt.savefig(save_path, dpi=300)
print(f"✅ Courbes générées et sauvegardées sous : {save_path}")
plt.show()