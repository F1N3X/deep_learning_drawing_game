# 🎮 DRAW IT!

Un jeu de dessin arcade inspiré de *Quick Draw* et *WarioWare* : dessine le mot affiché avant que le temps ne s'écoule, et le modèle d'IA devine ce que tu dessines **en temps réel**.

## Aperçu

* 15 classes à dessiner : avion, réveil, fourmi, pomme, hache, abeille, vélo, buisson, cactus, couronne, dauphin, dragon, pingouin, étoile, tour Eiffel
* 3 vies · chronomètre décroissant · événements WarioWare qui perturbent le canvas (flip, shake, blur…)
* Modèle CNN entraîné sur le dataset [Google Quick Draw](https://quickdraw.withgoogle.com/data), exporté en ONNX et servi via FastAPI

---

## Structure du projet

```text
.
├── ai/
│   ├── dataset/                  # Images d'entraînement (lancer dataset_generator.py pour l'obtenir)
│   ├── runs/                     # Logs TensorBoard
│   ├── best_model.pt             # Poids PyTorch du meilleur modèle
│   ├── dataset_generator.py      # Génération du dataset depuis les .ndjson
│   ├── deep_learning.ipynb       # Entraînement et évaluation du modèle
│   └── requirements.txt
├── backend/
│   ├── deep_learning_drawing_game.onnx   # Modèle exporté en ONNX
│   ├── main.py                   # API FastAPI (preprocessing + inférence)
│   └── requirements.txt
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── script.js             # Logique du jeu (canvas, timer, events)
│   │   ├── style.css             # UI rétro neon
│   │   └── assets/
│   ├── .env.example
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── quickdraw_ndjson/             # Accessibles à https://huggingface.co/datasets/Finesx/quickdraw_ndjson
└── README.md
```

---

## Installation

### Prérequis

* Python 3.10+
* Node.js 18+

## Backend

Créer et activer un environnement virtuel à la racine du projet :

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

Installer les dépendances du backend :

```bash
cd backend
pip install -r requirements.txt
```

Lancer l'API :

```bash
uvicorn main:app --reload
```

L'API est disponible sur `http://localhost:8000`.

## Frontend

Se placer dans le dossier du frontend :

```bash
cd frontend
```

Copier le fichier d'exemple :

```bash
cp .env.example .env
```

Sous Windows :

```powershell
copy .env.example .env
```

Modifier la variable contenant l'URL du backend si nécessaire. Par défaut, elle pointe vers l'API locale (`http://localhost:8000`).

Installer les dépendances et lancer le projet :

```bash
npm install
npm run dev
```

Le jeu est disponible sur `http://localhost:5173`.

---

## Modèle

Le CNN est entraîné sur ~1,9M d'images issues de Google Quick Draw, réparties en 15 classes.

| Split      |           Images |
| ---------- | ---------------: |
| Train      | 1 358 985 (70 %) |
| Test       |   291 211 (15 %) |
| Validation |   291 212 (15 %) |

### Architecture

```text
Conv2d(1→32) → BN → ReLU → MaxPool2d   # 64×64 → 32×32
Conv2d(32→64) → BN → ReLU → MaxPool2d  # 32×32 → 16×16
Flatten → Linear(16384→128) → ReLU → Dropout(0.2) → Linear(128→15)
```

### Prétraitement en inférence

1. Conversion en niveaux de gris + binarisation (seuil 128)
2. Crop sur le bounding box du tracé
3. Redimensionnement en conservant le ratio dans une image 64×64 (padding de 4 px)
4. Normalisation :

```python
(x / 255 - 0.5) / 0.5
```

---

## Générer le dataset

Télécharge les fichiers `.ndjson` correspondant aux 15 classes depuis le dataset Quick Draw et place-les dans `quickdraw_ndjson/`, puis :

```bash
cd ai
pip install -r requirements.txt

python dataset_generator.py
```

---

## Entraîner le modèle

Depuis le dossier `ai` :

```bash
pip install -r requirements.txt
```

Puis lancer le notebook :

```text
ai/deep_learning.ipynb
```

Le meilleur modèle est sauvegardé dans :

```text
ai/best_model.pt
```

---

## Exporter en ONNX

```python
import torch

model.load_state_dict(torch.load("best_model.pt"))

dummy = torch.randn(1, 1, 64, 64)

torch.onnx.export(
    model,
    dummy,
    "../backend/deep_learning_drawing_game.onnx",
    input_names=["input"],
    output_names=["output"]
)
```

---

## API

### `POST /predict`

#### Requête

```json
{
  "image": "<base64 PNG>"
}
```

#### Réponse

```json
{
  "guess": "cactus",
  "confidence": 94.2,
  "probabilities": {
    "cactus": 94.2,
    "airplane": 1.3
  }
}
```

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

## Règles du jeu

* Un mot apparaît : dessine-le sur le canvas blanc.
* Le modèle effectue une prédiction en temps réel toutes les 400 ms.
* Un dessin est validé si la prédiction correspond au mot demandé avec une confiance ≥ 85 %.
* Chaque vie réduit le temps disponible (20 s → 15 s → 10 s).
* Des événements aléatoires perturbent le canvas : flip horizontal/vertical, rotation 180°, shake, blur, obscurcissement, shrink, stripes.
* Score = `temps restant × 10` pour chaque mot trouvé.

---

## Stack technique

| Composant | Technologie                     |
| --------- | ------------------------------- |
| Modèle    | PyTorch → ONNX                  |
| Backend   | FastAPI + onnxruntime           |
| Frontend  | Vanilla JS + Vite               |
| UI        | CSS rétro néon (Press Start 2P) |
