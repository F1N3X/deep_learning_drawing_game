# 🎮 DRAW IT!

Un jeu de dessin arcade inspiré de *Quick Draw* et *WarioWare* : dessine le mot affiché avant que le temps ne s'écoule, et le modèle d'IA devine ce que tu dessines **en temps réel**.

## Aperçu

- 15 classes à dessiner : avion, réveil, fourmi, pomme, hache, abeille, vélo, buisson, cactus, couronne, dauphin, dragon, pingouin, étoile, tour Eiffel
- 3 vies · chronomètre décroissant · événements WarioWare qui perturbent le canvas (flip, shake, blur…)
- Modèle CNN entraîné sur le dataset [Google Quick Draw](https://quickdraw.withgoogle.com/data), exporté en ONNX et servi via FastAPI

---

## Structure du projet

```
.
├── ai/
│   ├── dataset/                  # Images d'entraînement (lancer dataset_generator.py pour l'obtenir)
│   ├── runs/                     # Logs TensorBoard
│   ├── best_model.pt             # Poids PyTorch du meilleur modèle
│   ├── dataset_generator.py      # Génération du dataset depuis les .ndjson
│   └── deep_learning.ipynb       # Entraînement et évaluation du modèle
├── backend/
│   ├── deep_learning_drawing_game.onnx   # Modèle exporté en ONNX
│   └── main.py                   # API FastAPI (preprocessing + inférence)
├── frontend/
│   ├── public/
│   └── src/
│       ├── script.js             # Logique du jeu (canvas, timer, events)
│       ├── style.css             # UI rétro neon
│       └── assets/
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── quickdraw_ndjson/             # Accessibles à https://huggingface.co/datasets/Finesx/quickdraw_ndjson
├── requirements.txt
└── README.md
```

---

## Installation

### Prérequis

- Python 3.10+
- Node.js 18+

### Backend

```bash
# Créer et activer l'environnement virtuel (à la racine du projet)
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

#Installer les dépendances
pip install -r requirements.txt

#  Lancer l'API
cd backend
uvicorn main:app --reload
```

L'API est disponible sur `http://localhost:8000`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Le jeu est disponible sur `http://localhost:5173`.

---

## Modèle

Le CNN est entraîné sur ~1,9M d'images issues de Google Quick Draw, réparties en 15 classes.

| Split | Images |
|-------|--------|
| Train | 1 358 985 (70 %) |
| Test | 291 211 (15 %) |
| Validation | 291 212 (15 %) |

**Architecture :**

```
Conv2d(1→32) → BN → ReLU → MaxPool2d   # 64×64 → 32×32
Conv2d(32→64) → BN → ReLU → MaxPool2d  # 32×32 → 16×16
Flatten → Linear(16384→128) → ReLU → Dropout(0.2) → Linear(128→15)
```

**Preprocessing inférence** (identique au dataset) :
1. Niveaux de gris + binarisation (seuil 128)
2. Crop sur le bounding box du tracé
3. Rescale avec ratio préservé dans 64×64 (padding 4px)
4. Normalisation : `(x / 255 - 0.5) / 0.5`

### Générer le dataset

Télécharge les fichiers `.ndjson` depuis [Quick Draw Dataset](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap) pour les 15 classes, place-les dans `quickdraw_ndjson/`, puis :

```bash
cd ai
python dataset_generator.py
```

### Entraîner le modèle

Lance le notebook `ai/deep_learning.ipynb` cellule par cellule. Le meilleur modèle est sauvegardé dans `ai/best_model.pt`.

### Exporter en ONNX

```python
import torch
model.load_state_dict(torch.load("best_model.pt"))
dummy = torch.randn(1, 1, 64, 64)
torch.onnx.export(model, dummy, "../backend/deep_learning_drawing_game.onnx",
                  input_names=["input"], output_names=["output"])
```

---

## API

### `POST /predict`

```json
// Request
{ "image": "<base64 PNG>" }

// Response
{
  "guess": "cactus",
  "confidence": 94.2,
  "probabilities": { "cactus": 94.2, "airplane": 1.3, ... }
}
```

### `GET /health`

```json
{ "status": "ok", "model_loaded": true }
```

---

## Règles du jeu

- Un mot apparaît : dessine-le sur le canvas blanc
- Le modèle prédit en temps réel (toutes les 400 ms)
- **Validé** si la prédiction correspond au mot avec ≥ 85 % de confiance
- Chaque vie donne moins de temps (20s → 15s → 10s)
- Des événements aléatoires perturbent le canvas : flip horizontal/vertical, rotation 180°, shake, blur, obscurcissement, shrink, stripes
- Score = `temps_restant × 10` par mot trouvé

---

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| Modèle | PyTorch → ONNX |
| Backend | FastAPI + onnxruntime |
| Frontend | Vanilla JS + Vite |
| UI | CSS neon rétro (Press Start 2P) |