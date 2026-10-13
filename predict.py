import tensorflow as tf
import numpy as np

MODEL_PATH = "best.keras"
IMG_PATH = r"C:\Users\souma\OneDrive\Documents\Alassane\IA2_EfficientNetB4\test_tomate\9.webp"
#IMG_PATH = r"C:\Users\souma\OneDrive\Documents\Alassane\IA2_EfficientNetB4\test_fraise\fraise1.jpg"
#IMG_PATH = r"C:\Users\souma\OneDrive\Documents\Alassane\IA2_EfficientNetB4\test_raisin\raisin_rust1.jpg"
#IMG_PATH = r"C:\Users\souma\OneDrive\Documents\Alassane\IA2_EfficientNetB4\test_mais\maisMalade.jpg"
#IMG_PATH = r"C:\Users\souma\OneDrive\Documents\Alassane\IA2_EfficientNetB4\test_citron\1.jpg"
#IMG_PATH = r"C:\Users\souma\OneDrive\Documents\Alassane\IA2_EfficientNetB4\test_papaye\1.jpg"
#IMG_PATH = r"C:\Users\souma\OneDrive\Documents\Alassane\IA2_EfficientNetB4\test_pomme_de_terre\3.webp"
IMG_SIZE = (380, 380)

model = tf.keras.models.load_model(MODEL_PATH)

# Les classes sont stockées dans le modèle via l’ordre des dossiers train.
# Pour être sûr, on recharge les classes depuis le train directory :
TRAIN_DIR = r"C:\Users\souma\OneDrive\Documents\Alassane\IA2_EfficientNetB4\dataset_split\train"
class_names = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMG_SIZE, batch_size=32, shuffle=False
).class_names

img = tf.keras.utils.load_img(IMG_PATH, target_size=IMG_SIZE)
x = tf.keras.utils.img_to_array(img)
x = np.expand_dims(x, axis=0)  # (1, 224, 224, 3)

probs = model.predict(x)[0]    # tableau de taille = nb_classes
idx = int(np.argmax(probs))

print("Classe prédite :", class_names[idx])
print("Confiance      :", float(probs[idx]))
if idx>0: 
    print("Classe prédite 2 :", class_names[idx-1])
    print("Confiance      :", float(probs[idx-1]))
if idx>1: 
    print("Classe prédite 3 :", class_names[idx-2])
    print("Confiance      :", float(probs[idx-2]))