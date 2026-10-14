from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import json
import io

app = Flask(__name__)

# -------------------------
# CONFIGURATION
# -------------------------
MODEL_PATH = "plant_disease_final_model.keras"
IMG_SIZE = (380, 380) # EfficientNetB4

print("Chargement du modèle...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Modèle chargé.")

# -------------------------
# CHARGEMENT CLASSES (ORDRE ALPHABÉTIQUE STRICT)
# -------------------------
try:
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    # L'entraînement trie les dossiers par ordre alphabétique.
    # On recrée cette liste exacte.
    class_names = sorted(list(class_indices.keys()))
    print(f"{len(class_names)} classes chargées (Tri alphabétique).")
except Exception as e:
    print(f"Erreur classes : {e}")
    class_names = []

# -------------------------
# CHARGEMENT RECOMMANDATIONS
# -------------------------
try:
    with open("recommendations.json", "r", encoding='utf-8') as f:
        recommendations = json.load(f)
except Exception:
    recommendations = {}

def get_advice_data(raw_name):
    if raw_name in recommendations: return recommendations[raw_name]
    norm1 = raw_name.replace("___", "_")
    if norm1 in recommendations: return recommendations[norm1]
    norm2 = raw_name.replace("___", " ").replace("_", " ")
    if norm2 in recommendations: return recommendations[norm2]
    return {"plant": "Inconnue", "advice": "Pas de conseil disponible."}

# -------------------------
# PRÉTRAITEMENT 100% TENSORFLOW
# -------------------------
def preprocess_image(image_bytes):
    """
    Utilise les fonctions TF pour être IDENTIQUE à l'entraînement.
    """
    # 1. Décodage de l'image brute (Jpg/Png) en Tensor
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    
    # 2. Redimensionnement EXACT de TensorFlow (Bilinear par défaut)
    img = tf.image.resize(img, IMG_SIZE)
    
    # 3. Conversion en float (pas de division par 255 car EfficientNet le fait en interne)
    # tf.image.resize renvoie déjà des floats, on s'assure juste du format.
    img = tf.cast(img, tf.float32)
    
    # 4. Ajout du batch (1, 380, 380, 3)
    img = tf.expand_dims(img, axis=0)
    
    return img

# -------------------------
# ROUTE API
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Aucune image reçue"}), 400

    try:
        file = request.files["image"]
        
        # On lit les bytes directement
        image_bytes = file.read()
        
        # Prétraitement via TensorFlow
        image_tensor = preprocess_image(image_bytes)

        # Prédiction
        preds = model.predict(image_tensor)
        
        # Résultats
        probs = preds[0]
        class_index = np.argmax(probs)
        confidence = float(probs[class_index])

        disease_raw = class_names[class_index]
        advice_data = get_advice_data(disease_raw)
        disease_display = disease_raw.replace("___", " ").replace("_", " ")

        print(f"Prédiction : {disease_display} | Confiance : {confidence:.4f}")

        return jsonify({
            "disease": disease_display,
            "confidence": confidence,
            "plant": advice_data["plant"],
            "advice": advice_data["advice"]
        })

    except Exception as e:
        print(f"Erreur : {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)