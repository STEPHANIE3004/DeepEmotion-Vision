# DeepEmotion-Vision 🎭
### Reconnaissance d'émotions faciales en temps réel — CNN + Transfer Learning (FER-2013)

---

## Démonstration

| 😢 **Tristesse — 68.8%** | 😊 **Joie — 21.6%** |
|:---:|:---:|
| ![Tristesse](https://raw.githubusercontent.com/STEPHANIE3004/DeepEmotion-Vision/main/screenshots/capture_20260410_132823.png) | ![Joie](https://raw.githubusercontent.com/STEPHANIE3004/DeepEmotion-Vision/main/screenshots/capture_20260410_132844.png) |

| 😨 **Peur — 43.6%** | 😡 **Colère — 61.4%** |
|:---:|:---:|
| ![Peur](https://raw.githubusercontent.com/STEPHANIE3004/DeepEmotion-Vision/main/screenshots/capture_20260410_132949.png) | ![Colère](https://raw.githubusercontent.com/STEPHANIE3004/DeepEmotion-Vision/main/screenshots/capture_20260410_133042.png) |

> Détection en temps réel via webcam avec panneau de probabilités pour les 7 émotions.

---

## Présentation du projet

Ce projet implémente un système de **reconnaissance d'émotions faciales en temps réel** à partir d'un flux webcam.
Un CNN entraîné sur le dataset **FER-2013** (35 887 images, 7 émotions) détecte et classe les expressions faciales frame par frame.

**Technologies clés :** TensorFlow/Keras · OpenCV · MediaPipe · Python 3.11

---

## Les 7 émotions détectées

| Émotion | Couleur d'affichage |
|---------|-------------------|
| 😡 Colère | Rouge |
| 🤢 Dégoût | Orange |
| 😨 Peur | Violet |
| 😊 Joie | Vert |
| 😢 Tristesse | Bleu foncé |
| 😲 Surprise | Jaune |
| 😐 Neutre | Gris |

---

## Fonctionnalités

- **Détection robuste** via MediaPipe Face Detection (fallback automatique sur Haar Cascade si absent)
- **Panneau de probabilités** en temps réel — 7 barres colorées affichées en live
- **Lissage temporel** — vote majoritaire sur 5 frames pour un affichage stable
- **Screenshot instantané** avec la touche `S` (sauvegarde dans `screenshots/`)
- **Deux approches de modélisation** : CNN from scratch + Transfer Learning MobileNetV2

---

## Architecture technique

### Modèle CNN (from scratch)

```
Input (48×48×1)
    → Bloc 1 : Conv2D(32) + BatchNorm + Conv2D(32) + BatchNorm + MaxPool + Dropout(0.25)
    → Bloc 2 : Conv2D(64) + BatchNorm + Conv2D(64) + BatchNorm + MaxPool + Dropout(0.25)
    → Bloc 3 : Conv2D(128) + BatchNorm + MaxPool + Dropout(0.25)
    → Flatten → Dense(256) + BatchNorm + Dropout(0.5)
    → Softmax(7)
```

### Transfer Learning (MobileNetV2)
- Base pré-entraînée sur ImageNet, gelée en Phase 1
- Fine-tuning des 30 dernières couches en Phase 2 (lr = 1e-4)
- Objectif : **65–70% accuracy** contre ~56% pour le CNN from scratch

---

## Modèle pré-entraîné inclus

Le fichier `mon_modele.keras` est directement inclus dans ce dépôt.

| Propriété | Valeur |
|-----------|--------|
| Format | Keras natif (`.keras`) |
| Taille | 4.2 MB |
| Dataset | FER-2013 |
| Epochs entraînés | 15 |
| Accuracy (test) | **55.89%** |
| Input | 48×48 pixels, niveaux de gris |
| Output | 7 classes d'émotions |
| Date d'entraînement | Février 2026 |

---

## Structure du projet

```
projet-vision-par-ordi/
├── code projet.ipynb       # Notebook : CNN from scratch + Transfer Learning MobileNetV2
├── demo_webcam.py          # Démo temps réel (webcam + panneau de probabilités)
├── mon_modele.keras        # Modèle CNN pré-entraîné (4.2 MB — prêt à l'emploi)
├── screenshots/            # Captures d'écran de la démo
│   ├── capture_20260410_132823.png   # Tristesse 68.8%
│   ├── capture_20260410_132844.png   # Joie 21.6%
│   ├── capture_20260410_132949.png   # Peur 43.6%
│   └── capture_20260410_133042.png   # Colère 61.4%
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/<votre-username>/deepemotion-vision.git
cd deepemotion-vision
```

### 2. Créer un environnement virtuel
```bash
python -m venv env

# Windows
env\Scripts\activate

# macOS / Linux
source env/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

---

## Utilisation

### Démonstration webcam (temps réel)
```bash
python demo_webcam.py
```

| Touche | Action |
|--------|--------|
| `Q` | Quitter |
| `S` | Sauvegarder un screenshot dans `screenshots/` |

### Entraîner le modèle

Ouvrez `code projet.ipynb` dans Jupyter ou PyCharm. Le notebook contient :

- **Sections 1–8** : CNN from scratch avec data augmentation, EarlyStopping, ReduceLROnPlateau
- **Section 9** : Transfer Learning MobileNetV2 (Phase 1 feature extraction + Phase 2 fine-tuning)

Structure attendue du dataset :
```
fer2013/
├── train/   (angry/ disgust/ fear/ happy/ sad/ surprise/ neutral/)
└── test/    (même structure)
```
Dataset disponible sur [Kaggle — FER-2013](https://www.kaggle.com/datasets/msambare/fer2013).

---

## Performances

| Approche | Accuracy (test) | Notes |
|----------|----------------|-------|
| CNN from scratch (15 epochs) | **55.89%** | Modèle inclus dans le repo |
| CNN + data augmentation + callbacks | ~58–62% | Avec EarlyStopping |
| Transfer Learning MobileNetV2 | ~65–70% | Section 9 du notebook |

> FER-2013 est un dataset difficile (labels bruités). L'état de l'art atteint ~71–75%.

---

## Technologies utilisées

- [TensorFlow / Keras](https://www.tensorflow.org/) — Deep learning
- [MediaPipe](https://mediapipe.dev/) — Détection de visage robuste
- [OpenCV](https://opencv.org/) — Traitement d'image et rendu vidéo
- [NumPy](https://numpy.org/) — Calcul numérique
- [Scikit-learn](https://scikit-learn.org/) — Métriques et évaluation
- [Matplotlib](https://matplotlib.org/) / [Seaborn](https://seaborn.pydata.org/) — Visualisation

---

## Auteure

**Stephanie Vanelle Mangoua** — Étudiante ingénieure à l'ESIEA
Projet réalisé dans le cadre du cours *Vision par Ordinateur*
