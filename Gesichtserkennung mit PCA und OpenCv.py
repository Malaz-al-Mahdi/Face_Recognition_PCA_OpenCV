"""
Gesichtserkennung mit PCA, Eigenfaces und OpenCV
=================================================
Dieses Programm implementiert eine PCA-basierte Gesichtserkennung (Eigenfaces).
Es extrahiert Gesichter aus Bildern, verarbeitet sie mit Hauptkomponentenanalyse (PCA)
und vergleicht sie mit einem bekannten Datensatz.

Autor: Malaz al Mahdi & Yasser Saadaoui
Institution: Goethe Universität

Quellen: 
- **OpenCV (Gesichtserkennung & Vorverarbeitung)**:
  - OpenCV Bibliothek: https://github.com/opencv/opencv

- **PCA-Implementierung für Eigenfaces**:
  - GitHub-Repository mit Beispielimplementierung: https://github.com/agyorev/Eigenfaces
  - GitHub-Repository zur Eigenface-Analyse: https://github.com/vutsalsinghal/EigenFace
  - GitHub-Repository für PCA-basierte Gesichtserkennung: https://github.com/zwChan/Face-recognition-using-eigenfaces/blob/master/eigenFace.py

- **Datenbank für Gesichtsbilder**:
  - https://github.com/vutsalsinghal/EigenFace/tree/master/Dataset

     Turk & Pentland (1991) - "Eigenfaces for Recognition"​

     Sirovich & Kirby (1987) - "Low-dimensional Procedure for the Characterization of Human Faces"
"""


import os
import math
import numpy as np
from matplotlib import pyplot as plt

import cv2


# ----------------------------------------------
# Konfiguration
# ----------------------------------------------

# Konstanten für die Bildgröße
WIDTH = 195
HEIGHT = 231

# Pfade zu den Trainings- und Testbildern
DATASET_PATH = 'Dataset/'
PROCESSED_PATH = 'Processed_Dataset/'  # Neuer Ordner für zugeschnittene Gesichter

# WICHTIG: Einheitlicher Schwellenwert
THRESHOLD = 7e7  

# Liste der Trainingsbilder
TRAIN_IMAGE_NAMES = [
    'subject01.normal.jpg', 'subject02.normal.jpg',
    'subject03.normal.jpg', 'subject07.normal.jpg',
    'subject10.normal.jpg', 'subject11.normal.jpg',
    'subject14.normal.jpg', 'subject15.normal.jpg'
]



# ----------------------------------------------
# 1) Funktion: Gesichtserkennung & Zuschneiden
# ----------------------------------------------
def detect_and_crop_faces(
    image_path,
    output_path,
    desired_width=195,
    desired_height=231,
    margin_percent=0.15,
    use_equalization=False,
    cascade_path='haarcascade_frontalface_alt2.xml'
):
  
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)
    
    # Bild laden
    image = cv2.imread(image_path)
    if image is None:
        print(f"Fehler beim Laden des Bildes: {image_path}")
        return None
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # In Graustufen umwandeln
    
    if use_equalization:
        gray_image = cv2.equalizeHist(gray_image)
    
    # Gesichtserkennung
    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.05,  # etwas engeres Scalen für genauere Erkennung
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(faces) == 0:
        print(f"Kein Gesicht gefunden in: {image_path}")
        return None
    
    # Nimm das erste erkannte Gesicht
    (x, y, w, h) = faces[0]
    
    # Margin berechnen
    margin_w = int(w * margin_percent)
    margin_h = int(h * margin_percent)
    
    # Neues (x, y, w, h) mit Rand
    x_new = max(x - margin_w, 0)
    y_new = max(y - margin_h, 0)
    w_new = min(w + 2 * margin_w, gray_image.shape[1] - x_new)
    h_new = min(h + 2 * margin_h, gray_image.shape[0] - y_new)
    
    # Gesicht zuschneiden
    cropped_face = gray_image[y_new:y_new+h_new, x_new:x_new+w_new]
    
    # Auf gewünschte Größe skalieren
    resized_face = cv2.resize(cropped_face, (desired_width, desired_height))
    
    # Gespeichertes Gesicht (optional)
    cv2.imwrite(output_path, resized_face)
    
    return resized_face


# ----------------------------------------------
# 2) Funktion: Trainingsbilder vorbereiten
# ----------------------------------------------

def prepare_training_images(image_names, dataset_path, processed_path, height, width):
    """
    Lädt die Trainingsbilder, erkennt das Gesicht, schneidet es zu und 
    wandelt es in einen Flattened-Vector um. Gibt ein Numpy-Array zurück.
    """
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    
    tensor = []
    for name in image_names:
        input_path = os.path.join(dataset_path, name)
        output_path = os.path.join(processed_path, name)
        
        face = detect_and_crop_faces(
            input_path, 
            output_path, 
            desired_width=width, 
            desired_height=height,
            margin_percent=0.15,
            use_equalization=False,
            cascade_path='haarcascade_frontalface_alt2.xml'
        )
        
        if face is None:
            print(f"Warnung: Kein Gesicht im Trainingsbild '{name}'.")
            continue
        
        tensor.append(face.flatten())
    
    return np.array(tensor)


# ----------------------------------------------
# 3) Funktion: Eigenes Hilfs-Display
# ----------------------------------------------

def display_images_grid(images, title, rows, cols, height, width, cmap='gray'):
    """
    Zeigt eine Reihe von 1D-Bildern (Flattened) in einem Gitter (rows x cols) an,
    jeweils reshaped auf (height x width).
    """
    plt.figure(figsize=(10, 5))
    plt.suptitle(title, fontsize=16, fontweight='bold')
    
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img.reshape(height, width), cmap=cmap)
        plt.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ----------------------------------------------
# 4) Funktionen: Durchschnitt & Normalisierung
# ----------------------------------------------

def calculate_mean_face(training_tensor):
    """Berechnet den Mittelwert über alle Trainingsbilder (Zeilen in training_tensor)."""
    return np.mean(training_tensor, axis=0)

def normalize_training_images(training_tensor, mean_face):
    """Subtrahiert das Durchschnittsgesicht von allen Trainingsvektoren."""
    return training_tensor - mean_face


# ----------------------------------------------
# 5) Eigenfaces berechnen
# ----------------------------------------------

def compute_eigenfaces(normalized_training_tensor):
    """
    Berechnet die Kovarianzmatrix und die dazugehörigen Eigenfaces.
    """
    cov_matrix = np.cov(normalized_training_tensor)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    print(eigenvalues)
    eigenfaces = np.dot(normalized_training_tensor.T, eigenvectors).T
    return eigenfaces


# ----------------------------------------------
# 6) Testbilder laden
# ----------------------------------------------

def load_test_images(dataset_path, processed_path, train_image_names, height, width):
    """
    Lädt alle Bilder, die NICHT in train_image_names enthalten sind.
    Gibt EINE Liste "test_data" zurück:

    
    - disp_img: immer (height,width) Graustufenbild (ggf. das Gesicht, 
                falls erkannt, sonst das verkleinerte Original)
    - face_vec: Flattened Face oder None
    """
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    
    all_files = os.listdir(dataset_path)
    test_data = []
    
    for name in all_files:
        # Überspringe Trainingsbilder
        if name in train_image_names:
            continue  
        
        input_path = os.path.join(dataset_path, name)
        output_path = os.path.join(processed_path, name)
        
        raw_img = cv2.imread(input_path)
        if raw_img is None:
            print(f"Fehler beim Laden des Bildes: {name} – wird übersprungen.")
            continue
        
        gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        
        # Versuche Face-Detection
        face_cropped = detect_and_crop_faces(
            input_path,
            output_path,
            desired_width=width,
            desired_height=height,
            margin_percent=0.15,
            use_equalization=False,
            cascade_path='haarcascade_frontalface_alt2.xml'
        )
        
        if face_cropped is None:
            # => Kein Gesicht -> disp_img = verkleinerte Originalversion
            disp_img = cv2.resize(gray_img, (width, height))
            face_vec = None
        else:
            # => Gesicht erkannt
            disp_img = face_cropped  # Hat schon (height,width)
            face_vec = face_cropped.flatten()
        
        test_data.append((disp_img, face_vec, name))
    
    return test_data


# ----------------------------------------------
# 7) Gesichtserkennung mit Schwellenwert
# ----------------------------------------------

def filter_unrecognized_faces(
    test_face,
    mean_face,
    eigenfaces,
    training_tensor,
    train_image_names,
    threshold=7e7
):
    """
    Vergleicht test_face (Flattened) mit den Trainingsbildern und 
    entscheidet anhand eines Schwellenwerts, ob das Bild bekannt ist.
    Gibt (is_unknown, distances) oder (False, distances, min_idx) zurück.
    """
    # Normalisieren
    normalized_test_image = test_face - mean_face
    
    # Eigenfaces-Gewichte für das Testbild
    weights = np.dot(eigenfaces, normalized_test_image)
    
    # Eigenfaces-Gewichte für alle Trainingsbilder
    training_weights = np.dot(eigenfaces, (training_tensor - mean_face).T).T
    
    # Euklidische Distanzen
    distances = np.linalg.norm(training_weights - weights, axis=1)
    
    # Minimale Distanz
    min_distance = np.min(distances)
    min_index = np.argmin(distances)
    
    # Distanzen ausgeben
    print(f"Distanzen des Testbilds zu den Trainingsbildern:")
    for name, dist in zip(train_image_names, distances):
        print(f"{name}: {dist:.2f}")
    
    # Vergleich mit threshold
    if min_distance >= threshold:
        print(f"Unbekanntes Gesicht: Min. Distanz: {min_distance:.2f} (>= {threshold:.2f})")
        return True, distances
    else:
        print(f"Erkannt als: {train_image_names[min_index]} mit Distanz: {min_distance:.2f}")
        return False, distances, min_index

# ----------------------------------------------
# Funktion: Erkannte Gesichter anzeigen
# ----------------------------------------------

def display_recognized_faces(recognized_faces):
    """
    Zeigt alle erfolgreich erkannten Gesichter mit ihrem zugeordneten Namen an.
    :param recognized_faces: Liste von Tupeln (Gesichtsbild, erkannter Name)
    """
    if len(recognized_faces) == 0:
        print("Keine Gesichter wurden erkannt.")
        return

    # Anzahl der erkannten Gesichter
    num_faces = len(recognized_faces)
    cols = 4  # Anzahl der Spalten
    rows = math.ceil(num_faces / cols)

    # Fenster zur Anzeige der erkannten Gesichter
    plt.figure(figsize=(15, 10))
    plt.suptitle("Erkannte Gesichter", fontsize=16, fontweight='bold')

    for i, (img, name) in enumerate(recognized_faces):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img.reshape(HEIGHT, WIDTH), cmap='gray')
        plt.title(name, fontsize=8)  # Zeigt den Namen des erkannten Gesichts
        plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ----------------------------------------------
# Hauptprogramm
# ----------------------------------------------
if __name__ == "__main__":
    
    # 1) Trainingsbilder vorbereiten
    training_tensor = prepare_training_images(
        TRAIN_IMAGE_NAMES, 
        DATASET_PATH, 
        PROCESSED_PATH, 
        HEIGHT, 
        WIDTH
    )
    
    # Anzeige: Trainingsbilder
    display_images_grid(training_tensor, "Trainingsbilder", 2, 4, HEIGHT, WIDTH)
    
    # 2) Durchschnittsgesicht
    mean_face = calculate_mean_face(training_tensor)
    display_images_grid([mean_face], "Durchschnittsgesicht", 1, 1, HEIGHT, WIDTH)
    
    # 3) Normalisierte Gesichter
    normalized_training_tensor = normalize_training_images(training_tensor, mean_face)
    display_images_grid(normalized_training_tensor, "Normalisierte Gesichter", 2, 4, HEIGHT, WIDTH)
    
    # 4) Eigenfaces
    eigenfaces = compute_eigenfaces(normalized_training_tensor)
    # Zeige nur die ersten 8 Eigenfaces
    display_images_grid(eigenfaces[:8], "Eigenfaces", 2, 4, HEIGHT, WIDTH, cmap='jet')
    
    # 5) Testbilder laden (alle, die NICHT in TRAIN_IMAGE_NAMES sind)
    test_data = load_test_images(
        DATASET_PATH, 
        PROCESSED_PATH,
        TRAIN_IMAGE_NAMES,
        HEIGHT, 
        WIDTH
    )
    
    # Anzeige aller Testbilder in einem Raster
    def display_test_data(test_data, title="Testbilder", cmap='gray', cols=4):
        num_images = len(test_data)
        rows = math.ceil(num_images / cols)
        
        plt.figure(figsize=(10, 5))
        plt.suptitle(title, fontsize=16, fontweight='bold')
        
        for i, (img, face_vec, name) in enumerate(test_data):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img, cmap=cmap)
            plt.title(name, fontsize=8)  # optionaler Dateiname im Titel
            plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    
    # Einmalig zeigen
    display_test_data(test_data, "Testbilder", cmap='gray', cols=4)
    
# 6) Gesichtserkennung
recognized_faces = []  # Liste für erkannte Gesichter
unrecognized_images = []  # Liste für unbekannte oder gesichtslose Bilder

for (disp_img, face_vec, filename) in test_data:
    print(f"\nErgebnisse für Testbild: {filename}")
    
    # Fall A: Kein Gesicht erkannt
    if face_vec is None:
        print("=> Kein Gesicht erkannt in diesem Bild!")
        # min_dist = None => Symbolisiert "kein Gesicht"
        unrecognized_images.append((disp_img, filename, None))
        continue
    
    # Fall B: face_vec existiert -> versuche PCA-Vergleich
    result = filter_unrecognized_faces(
        face_vec,
        mean_face,
        eigenfaces,
        training_tensor,
        TRAIN_IMAGE_NAMES,
        threshold=THRESHOLD
    )
    
    if result[0]:  # => Unbekannt
        _, distances = result
        min_dist = np.min(distances)
        unrecognized_images.append((disp_img, filename, min_dist))
    else:
        # => Erkannt
        _, distances, min_index = result
        recognized_faces.append((disp_img, TRAIN_IMAGE_NAMES[min_index]))  # Gesicht und Name speichern
        print(f"{filename} erkannt als {TRAIN_IMAGE_NAMES[min_index]} "
              f"mit Distanz {distances[min_index]:.2f}")

# 7) Zeige nur unbekannte / ohne Gesicht in EINEM Raster
if len(unrecognized_images) > 0:
    num_unrec = len(unrecognized_images)
    cols = 4
    rows = math.ceil(num_unrec / cols)
    
    plt.figure(figsize=(10, 5))
    plt.suptitle("Unbekannte oder kein Gesicht", fontsize=16, fontweight='bold')
    
    for i, (img, name, min_dist) in enumerate(unrecognized_images):
        plt.subplot(rows, cols, i+1)
        plt.imshow(img, cmap='gray')
        if min_dist is None:
            # => Kein Gesicht
            plt.title(f"Kein Gesicht:\n{name}", fontsize=8)
        else:
            plt.title(f"Unbekannt:\n{name}\nDist={min_dist:.2f}", fontsize=8)
        plt.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
else:
    print("Es wurden keine unbekannten Gesichter bzw. Bilder ohne erkanntes Gesicht gefunden.")

# 8) Zeige erkannte Gesichter
display_recognized_faces(recognized_faces)
