import os
import io
import time
import base64
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
import requests
import onnxruntime as ort
from PIL import Image
from cachetools import TTLCache
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# CORS via variável de ambiente
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("BACKEND_ALLOWED_ORIGINS", "").split(",") if o.strip()]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["*"]  # uso local/desenvolvimento

CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}}, supports_credentials=True)

# Pasta local de modelos
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ArcFace R100 ONNX (oficial InsightFace release)
ARCFACE_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/arcface_r100_v1.onnx"
ARCFACE_PATH = os.path.join(MODELS_DIR, "arcface_r100_v1.onnx")

# Cache de embeddings de motoristas: 5min, até 256 itens
driver_embedding_cache: TTLCache[str, np.ndarray] = TTLCache(maxsize=256, ttl=300)

# Detector Haar (não precisa compilar nada)
HAAR_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(HAAR_PATH)

# Sessão ONNXRuntime (CPU)
_arcface_sess: Optional[ort.InferenceSession] = None
_arcface_input_name: Optional[str] = None
_arcface_output_name: Optional[str] = None


def ensure_arcface_model() -> None:
    if not os.path.exists(ARCFACE_PATH):
        logging.info("Baixando modelo ArcFace ONNX...")
        r = requests.get(ARCFACE_URL, timeout=60)
        r.raise_for_status()
        with open(ARCFACE_PATH, "wb") as f:
            f.write(r.content)
        logging.info("Modelo ArcFace salvo em %s", ARCFACE_PATH)


def get_arcface_session() -> ort.InferenceSession:
    global _arcface_sess, _arcface_input_name, _arcface_output_name
    if _arcface_sess is None:
        ensure_arcface_model()
        so = ort.SessionOptions()
        _arcface_sess = ort.InferenceSession(ARCFACE_PATH, sess_options=so, providers=["CPUExecutionProvider"])
        _arcface_input_name = _arcface_sess.get_inputs()[0].name
        _arcface_output_name = _arcface_sess.get_outputs()[0].name
    return _arcface_sess


def b64_to_rgb(image_data_uri: str) -> np.ndarray:
    try:
        if "," in image_data_uri:
            _, enc = image_data_uri.split(",", 1)
        else:
            enc = image_data_uri
        raw = base64.b64decode(enc)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return np.array(img)
    except Exception as e:
        raise ValueError(f"Imagem base64 inválida: {e}")


def detect_face_bbox(rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    # Haar opera em escala de cinza
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    # pega o maior rosto
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    # margem (20%) para melhorar enquadramento
    m = int(0.2 * max(w, h))
    x0 = max(0, x - m)
    y0 = max(0, y - m)
    x1 = min(rgb.shape[1], x + w + m)
    y1 = min(rgb.shape[0], y + h + m)
    return (x0, y0, x1 - x0, y1 - y0)


def preprocess_arcface(rgb_face: np.ndarray) -> np.ndarray:
    # ArcFace espera 112x112, normalizado para [-1, 1], CHW
    img = cv2.resize(rgb_face, (112, 112), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0).copy()
    return img


def extract_embedding(rgb: np.ndarray) -> Optional[np.ndarray]:
    bbox = detect_face_bbox(rgb)
    if bbox is None:
        return None
    x, y, w, h = bbox
    face = rgb[y : y + h, x : x + w]
    if face.size == 0:
        return None

    inp = preprocess_arcface(face)
    sess = get_arcface_session()
    emb = sess.run([_arcface_output_name], { _arcface_input_name: inp })[0].squeeze()
    # normaliza (L2) para estabilidade na similaridade cosseno
    emb = emb / (norm(emb) + 1e-12)
    return emb.astype(np.float32)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2) / (norm(v1) * norm(v2) + 1e-12))


def get_driver_embeddings(drivers: List[dict]) -> Tuple[List[np.ndarray], List[Tuple[str, str]]]:
    known_embeddings: List[np.ndarray] = []
    id_name_pairs: List[Tuple[str, str]] = []

    # Limpa do cache IDs que não vieram na lista atual
    incoming = {d.get("id") for d in drivers if d.get("id")}
    for k in list(driver_embedding_cache.keys()):
        if k not in incoming:
            driver_embedding_cache.pop(k, None)

    for d in drivers:
        driver_id = d.get("id")
        name = d.get("name")
        photo_url = d.get("photo_url")
        if not (driver_id and name and photo_url):
            logging.warning(f"Dados de motorista incompletos: {d}")
            continue

        emb = driver_embedding_cache.get(driver_id)
        if emb is None:
            try:
                logging.info(f"Baixando foto de {name} ({driver_id})...")
                r = requests.get(photo_url, timeout=15)
                r.raise_for_status()
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
                emb = extract_embedding(np.array(img))
                if emb is not None:
                    driver_embedding_cache[driver_id] = emb
                    logging.info(f"Embedding cacheado para {name} ({driver_id}).")
                else:
                    logging.warning(f"Nenhum rosto detectado na foto de {name} ({driver_id}).")
                    continue
            except Exception as e:
                logging.error(f"Erro ao processar {name} ({driver_id}): {e}")
                continue

        known_embeddings.append(emb)
        id_name_pairs.append((driver_id, name))

    return known_embeddings, id_name_pairs


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "API de reconhecimento facial (ArcFace ONNX) online!"})


@app.route("/verify_driver", methods=["POST"])
def verify_driver():
    start = time.perf_counter()
    payload = request.get_json(silent=True) or {}

    image_data = payload.get("image")
    authorized_drivers = payload.get("authorized_drivers", [])
    car_id = payload.get("car_id", "N/A")

    if not image_data or not isinstance(authorized_drivers, list):
        return jsonify({"error": "Requisição inválida. Envie 'image' (base64) e 'authorized_drivers' (lista)."}), 400

    logging.info(f"Verificação do carro {car_id} com {len(authorized_drivers)} motoristas...")

    try:
        rgb = b64_to_rgb(image_data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    query_emb = extract_embedding(rgb)
    if query_emb is None:
        return jsonify({
            "authorized": False,
            "message": "Nenhum rosto detectado na captura.",
            "driver_id": None,
            "driver_name": "Desconhecido",
            "confidence": 0,
            "processing_time": round(time.perf_counter() - start, 2),
        }), 200

    known_embeddings, id_name_pairs = get_driver_embeddings(authorized_drivers)
    if not known_embeddings:
        return jsonify({
            "authorized": False,
            "message": "Nenhum motorista autorizado processado no backend.",
            "driver_id": None,
            "driver_name": "Desconhecido",
            "confidence": 0,
            "processing_time": round(time.perf_counter() - start, 2),
        }), 200

    sims = [cosine_similarity(query_emb, emb) for emb in known_embeddings]
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    best_driver_id, best_driver_name = id_name_pairs[best_idx]

    # Limiar típico para ArcFace sem alinhamento perfeito (ajuste conforme necessário)
    TH_STRICT = 0.55
    TH_LOOSE = 0.48

    authorized = best_sim >= TH_LOOSE
    if best_sim >= TH_STRICT:
        message = f"Motorista {best_driver_name} autorizado (alta confiança)"
    elif best_sim >= TH_LOOSE:
        message = f"Motorista {best_driver_name} autorizado (confiança moderada)"
    else:
        message = "Motorista não reconhecido ou não autorizado"

    return jsonify({
        "authorized": authorized,
        "message": message,
        "driver_id": best_driver_id if authorized else None,
        "driver_name": best_driver_name if authorized else "Desconhecido",
        "confidence": round(best_sim * 100, 1),
        "processing_time": round(time.perf_counter() - start, 2),
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logging.info(f"Iniciando servidor Flask na porta {port}...")
    app.run(debug=True, host="0.0.0.0", port=port)