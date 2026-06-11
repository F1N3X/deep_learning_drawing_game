from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
import onnxruntime as ort
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = ["The Eiffel Tower","airplane", "alarm clock", "ant", "apple", "axe","bee", "bicycle", "bush", "cactus", "crown","dolphin", "dragon", "penguin", "star"]

session = None

@app.on_event("startup")
def load_model():
    global session
    try:
        session = ort.InferenceSession("deep_learning_drawing_game.onnx")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")

class ImagePayload(BaseModel):
    image: str  # base64 encoded PNG from canvas

def preprocess(img: Image.Image) -> np.ndarray:
    """
    Reproduit exactement le preprocessing de dataset_generator.py :
    1. Convertir en niveaux de gris
    2. Binariser (seuil 128) pour isoler le trait proprement
    3. Trouver le bounding box du dessin
    4. Recadrer sur le bounding box
    5. Rescaler en gardant le ratio, avec padding=4 sur 64px
    6. Centrer dans un canvas 64×64 blanc
    7. Normaliser : ÷255, puis (x - 0.5) / 0.5
    """
    IMG_SIZE = 64
    PADDING  = 4

    # 1. Grayscale
    img = img.convert("L")
    arr = np.array(img, dtype=np.uint8)

    # 2. Binarize — seuil à 128 : noir=0, blanc=255
    binary = np.where(arr < 128, 0, 255).astype(np.uint8)

    # 3. Bounding box des pixels noirs (valeur 0)
    rows = np.any(binary == 0, axis=1)
    cols = np.any(binary == 0, axis=0)

    if not rows.any():
        # Canvas vide → retourner image blanche normalisée
        blank = np.ones((1, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        return (blank - 0.5) / 0.5

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # 4. Crop sur le bounding box
    cropped = binary[rmin:rmax+1, cmin:cmax+1]

    h, w = cropped.shape
    width  = max(w, 1)
    height = max(h, 1)

    # 5. Scale pour tenir dans (IMG_SIZE - 2*PADDING) en gardant le ratio
    scale = min(
        (IMG_SIZE - 2 * PADDING) / width,
        (IMG_SIZE - 2 * PADDING) / height
    )

    new_w = max(int(width  * scale), 1)
    new_h = max(int(height * scale), 1)

    resized = np.array(
        Image.fromarray(cropped).resize((new_w, new_h), Image.LANCZOS)
    )

    # 6. Centrer dans canvas 64×64 blanc
    canvas = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)
    offset_y = (IMG_SIZE - new_h) // 2
    offset_x = (IMG_SIZE - new_w) // 2
    canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized

    # 7. Normaliser : ToTensor (÷255) puis Normalize(mean=0.5, std=0.5)
    arr_f = canvas.astype(np.float32) / 255.0
    arr_f = (arr_f - 0.5) / 0.5

    # Shape [1, 1, 64, 64]
    return arr_f[np.newaxis, np.newaxis, :, :]


@app.post("/predict")
def predict(payload: ImagePayload):
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        img_data = payload.image
        if "," in img_data:
            img_data = img_data.split(",")[1]

        raw = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(raw))

        arr = preprocess(img)

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: arr})
        logits = outputs[0][0]  # [15]

        # Softmax
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()

        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx]) * 100

        return {
            "guess": CLASSES[top_idx],
            "confidence": round(confidence, 1),
            "probabilities": {
                CLASSES[i]: round(float(probs[i]) * 100, 1)
                for i in range(len(CLASSES))
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": session is not None}