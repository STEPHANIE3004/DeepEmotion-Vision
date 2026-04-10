import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import os
from datetime import datetime

# ==========================================================
# 1. CONFIGURATION
# ==========================================================

# noinspection SpellCheckingInspection
MODEL_PATH = "mon_modele.keras"
EMOTIONS   = ['Colere', 'Degout', 'Peur', 'Joie', 'Triste', 'Surprise', 'Neutre']

# Couleur associée à chaque émotion (BGR)
COLORS = [
    (0,   0,   220),   # Colère   → Rouge
    (0,   140, 255),   # Dégoût   → Orange
    (200, 0,   200),   # Peur     → Violet
    (0,   210, 0  ),   # Joie     → Vert
    (220, 80,  0  ),   # Triste   → Bleu foncé
    (0,   220, 220),   # Surprise → Jaune
    (160, 160, 160),   # Neutre   → Gris
]

# ==========================================================
# 2. DÉTECTION DE VISAGES : MediaPipe ou Haar Cascade
# ==========================================================

# On initialise à None pour éviter les avertissements "can be undefined"
mp_face_detection = None
face_detection    = None
face_cascade      = None
USE_MEDIAPIPE     = False

try:
    from mediapipe.solutions.face_detection import FaceDetection as _MpFaceDetection
    face_detection = _MpFaceDetection(min_detection_confidence=0.5)
    USE_MEDIAPIPE  = True
    print("MediaPipe chargé — détection de visages améliorée.")
except (ImportError, AttributeError):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # noqa
    )
    print("MediaPipe incompatible — utilisation du Haar Cascade (fallback).")

# ==========================================================
# 3. CHARGEMENT DU MODÈLE
# ==========================================================

model = load_model(MODEL_PATH)

# Lissage temporel : vote majoritaire sur 5 frames
historique_emotions = deque(maxlen=5)

# Dossier de sauvegarde des screenshots
os.makedirs("screenshots", exist_ok=True)


# ==========================================================
# 4. FONCTION : PANNEAU DE PROBABILITÉS
# ==========================================================

def draw_probability_bars(image, probs, labels, palette):
    """Affiche un panneau de barres de probabilités à droite de l'image."""
    img_h, img_w = image.shape[:2]
    panel_x   = img_w - 230
    bar_h     = 22
    bar_max_w = 180
    margin    = 7
    panel_top = 5
    panel_bot = panel_top + len(labels) * (bar_h + margin) + 10

    # Fond semi-transparent
    overlay = image.copy()
    cv2.rectangle(overlay, (panel_x - 8, panel_top), (img_w - 4, panel_bot), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)

    for i, (label, prob) in enumerate(zip(labels, probs)):
        bar_y     = panel_top + 8 + i * (bar_h + margin)
        bar_width = int(prob * bar_max_w)

        # Barre de fond
        cv2.rectangle(image, (panel_x, bar_y),
                      (panel_x + bar_max_w, bar_y + bar_h), (55, 55, 55), -1)
        # Barre remplie
        if bar_width > 0:
            cv2.rectangle(image, (panel_x, bar_y),
                          (panel_x + bar_width, bar_y + bar_h), palette[i], -1)
        # Texte
        cv2.putText(image,
                    f"{label}: {prob * 100:.1f}%",
                    (panel_x + 4, bar_y + bar_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


# ==========================================================
# 5. FLUX VIDÉO EN TEMPS RÉEL
# ==========================================================

cap = cv2.VideoCapture(0)
print(f"Modèle '{MODEL_PATH}' chargé avec succès.")
print("Touches : [Q] Quitter  |  [S] Screenshot")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_coords = []  # Liste de tuples (x, y, x2, y2)

    # --- Détection des visages ---
    if USE_MEDIAPIPE and face_detection is not None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = face_detection.process(rgb_frame)
        if results.detections:
            for det in results.detections:
                bb  = det.location_data.relative_bounding_box
                x   = max(0, int(bb.xmin * frame_w))
                y   = max(0, int(bb.ymin * frame_h))
                x2  = min(frame_w, int((bb.xmin + bb.width)  * frame_w))
                y2  = min(frame_h, int((bb.ymin + bb.height) * frame_h))
                faces_coords.append((x, y, x2, y2))
    elif face_cascade is not None:
        raw = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (fx, fy, fw, fh) in raw:
            faces_coords.append((fx, fy, fx + fw, fy + fh))

    current_probs = None

    for (x, y, x2, y2) in faces_coords:
        roi = gray[y:y2, x:x2]
        if roi.size == 0:
            continue

        # Prétraitement → 48×48, normalisé
        roi = cv2.resize(roi, (48, 48)).astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=-1)
        roi = np.expand_dims(roi, axis=0)

        # Inférence
        prediction    = model.predict(roi, verbose=0)
        current_probs = prediction[0]
        index_actuel  = int(np.argmax(current_probs))

        # Lissage temporel
        historique_emotions.append(index_actuel)
        index_stable  = max(set(historique_emotions),
                            key=list(historique_emotions).count)

        emotion_label = EMOTIONS[index_stable]
        confiance     = current_probs[index_actuel] * 100
        color         = COLORS[index_stable]

        # Rectangle autour du visage
        cv2.rectangle(frame, (x, y), (x2, y2), color, 2)

        # Bandeau au-dessus du visage
        label_text = f"{emotion_label}  {confiance:.1f}%"
        label_w    = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
        cv2.rectangle(frame, (x, y - 32), (x + label_w + 10, y), color, -1)
        cv2.putText(frame, label_text, (x + 5, y - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Panneau de probabilités (si au moins un visage détecté)
    if current_probs is not None:
        draw_probability_bars(frame, current_probs, EMOTIONS, COLORS)

    # Indication des touches
    cv2.putText(frame, "[Q] Quitter  |  [S] Screenshot",
                (10, frame_h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.imshow("DeepEmotion-Vision - Real Time", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshots/capture_{ts}.png"
        cv2.imwrite(filename, frame)
        print(f"Screenshot sauvegardé : {filename}")

# ==========================================================
# 6. FERMETURE PROPRE
# ==========================================================
cap.release()
cv2.destroyAllWindows()
if USE_MEDIAPIPE and face_detection is not None:
    face_detection.close()
