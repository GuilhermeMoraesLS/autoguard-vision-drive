import os
import io
import time
import base64
import logging
import traceback  # ADICIONE ESTA LINHA
from typing import List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image
from cachetools import TTLCache
from flask import Flask, request, jsonify
from flask_cors import CORS
from numpy.linalg import norm  # ADICIONE ESTA LINHA TAMBÉM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# CORS via variável de ambiente
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("BACKEND_ALLOWED_ORIGINS", "").split(",") if o.strip()]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["*"]  # uso local/desenvolvimento

CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}}, supports_credentials=True)

# Pasta local de modelos
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Cache de embeddings de motoristas: 5min, até 256 itens
driver_embedding_cache: TTLCache[str, np.ndarray] = TTLCache(maxsize=256, ttl=300)

# Detector Haar (não precisa compilar nada)
HAAR_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(HAAR_PATH)


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


# Adicione esta função antes de extract_embedding:
def assess_image_quality(face_gray: np.ndarray) -> float:
    """Avalia a qualidade da imagem facial (0-1, maior é melhor)"""
    # Verifica brilho adequado
    mean_brightness = np.mean(face_gray)
    brightness_score = 1.0 - abs(mean_brightness - 128) / 128
    
    # Verifica contraste
    contrast_score = min(np.std(face_gray) / 50, 1.0)
    
    # Verifica nitidez (Laplaciano)
    laplacian_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()
    sharpness_score = min(laplacian_var / 500, 1.0)
    
    # Score combinado
    quality = (brightness_score + contrast_score + sharpness_score) / 3
    return quality


# Substitua a função extract_embedding por esta versão corrigida:
def extract_embedding(rgb: np.ndarray) -> Optional[np.ndarray]:
    bbox = detect_face_bbox(rgb)
    if bbox is None:
        return None
    
    x, y, w, h = bbox
    face = rgb[y : y + h, x : x + w]
    if face.size == 0:
        return None
    
    # Verifica qualidade mínima da imagem
    face_gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    quality = assess_image_quality(face_gray)
    logging.info(f"🎯 Qualidade da imagem: {quality:.3f}")
    
    if quality < 0.3:  # Limiar mínimo de qualidade
        logging.warning("⚠️ Qualidade de imagem muito baixa")
        return None
    
    # Normaliza o tamanho da face para análise consistente
    face_norm = cv2.resize(face, (128, 128))
    face_gray_norm = cv2.cvtColor(face_norm, cv2.COLOR_RGB2GRAY)
    
    # Aplica equalização de histograma para melhorar contraste
    face_eq = cv2.equalizeHist(face_gray_norm)
    
    # Divide a face em múltiplas regiões para análise detalhada
    regions = []
    grid_size = 8  # Grid 8x8 = 64 regiões
    cell_h = 128 // grid_size
    cell_w = 128 // grid_size
    
    for i in range(grid_size):
        for j in range(grid_size):
            y_start = i * cell_h
            y_end = (i + 1) * cell_h
            x_start = j * cell_w
            x_end = (j + 1) * cell_w
            
            cell = face_eq[y_start:y_end, x_start:x_end]
            
            if cell.size == 0:
                continue
                
            # Múltiplas características por célula
            regions.extend([
                float(np.mean(cell)),                    # Brilho médio
                float(np.std(cell)),                     # Variação
                float(np.min(cell)),                     # Valor mínimo
                float(np.max(cell)),                     # Valor máximo
                float(np.median(cell)),                  # Mediana
                float(np.percentile(cell, 25)),          # 1º quartil
                float(np.percentile(cell, 75)),          # 3º quartil
                float(np.sum(np.gradient(cell.astype(float))[0]**2) + np.sum(np.gradient(cell.astype(float))[1]**2)), # Textura/bordas
            ])
    
    # Calcula gradientes FORA da lista
    grad_x = cv2.Sobel(face_gray_norm, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(face_gray_norm, cv2.CV_64F, 0, 1, ksize=3)
    
    # Características globais da face
    global_features = [
        # Estatísticas gerais
        float(np.mean(face_gray_norm)),
        float(np.std(face_gray_norm)),
        float(np.median(face_gray_norm)),
        float(np.var(face_gray_norm)),
        
        # Distribuição de intensidades
        float(np.sum(face_gray_norm < 50)),   # Pixels escuros
        float(np.sum(face_gray_norm > 200)),  # Pixels claros
        float(np.sum((face_gray_norm >= 50) & (face_gray_norm <= 200))),  # Pixels médios
        
        # Características de bordas (Sobel)
        float(np.sum(grad_x**2)),  # Bordas horizontais
        float(np.sum(grad_y**2)),  # Bordas verticais
        
        # Características de textura (Laplaciano)
        float(np.sum(cv2.Laplacian(face_gray_norm, cv2.CV_64F)**2)),
        
        # Simetria facial (comparação esquerda vs direita)
        float(np.mean(np.abs(face_gray_norm[:, :64] - np.fliplr(face_gray_norm[:, 64:])))),
        
        # Direção média dos gradientes
        float(np.mean(np.arctan2(grad_y, grad_x + 1e-8))),
        
        # Características de regiões específicas (aproximadas)
        float(np.mean(face_gray_norm[20:40, 40:88])),   # Região dos olhos
        float(np.mean(face_gray_norm[60:80, 50:78])),   # Região do nariz  
        float(np.mean(face_gray_norm[85:105, 45:83])),  # Região da boca
    ]
    
    # Combina todas as características
    embedding = np.array(regions + global_features, dtype=np.float32)
    
    # Aplica PCA mock (reduz correlação entre características)
    embedding_unique = []
    for i in range(0, len(embedding), 4):
        chunk = embedding[i:i+4]
        # Combinação linear para reduzir correlação
        if len(chunk) == 4:
            embedding_unique.extend([
                float(chunk[0] - chunk[1]),              # Diferença
                float(chunk[2] * chunk[3]),              # Produto
                float(np.mean(chunk)),                   # Média
                float(np.std(chunk)),                    # Desvio
            ])
        else:
            embedding_unique.extend([float(x) for x in chunk])
    
    # Padding/truncate para exatamente 512 dimensões
    embedding_final = np.array(embedding_unique, dtype=np.float32)
    
    # Adiciona características aleatórias se necessário
    while len(embedding_final) < 512:
        additional_features = []
        for i in range(min(100, 512 - len(embedding_final))):
            # Adiciona características derivadas existentes com ruído
            if len(embedding_unique) > 0:
                base_idx = i % len(embedding_unique)
                noise = np.random.normal(0, 0.001)
                additional_features.append(float(embedding_unique[base_idx] + noise))
            else:
                additional_features.append(float(np.random.normal(0, 0.1)))
        
        embedding_final = np.concatenate([embedding_final, additional_features])
    
    # Trunca para exatamente 512 dimensões
    embedding_final = embedding_final[:512]
    
    # Normalização robusta
    mean_emb = np.mean(embedding_final)
    std_emb = np.std(embedding_final)
    if std_emb > 1e-8:
        embedding_final = (embedding_final - mean_emb) / std_emb
    
    # Normalização L2 final
    norm_val = np.linalg.norm(embedding_final)
    if norm_val > 1e-8:
        embedding_final = embedding_final / norm_val
    
    return embedding_final.astype(np.float32)


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
    logging.info("🔄 Requisição recebida em /verify_driver")
    logging.info(f"📊 Headers: {dict(request.headers)}")
    logging.info(f"🌐 Origem da requisição: {request.headers.get('Origin', 'N/A')}")
    logging.info(f"📝 Content-Type: {request.headers.get('Content-Type', 'N/A')}")
    
    try:
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
        # Limiares muito mais restritivos para melhor discriminação
        TH_STRICT = 0.95   # Aumentado de 0.85 para 0.95 (muito restritivo)
        TH_LOOSE = 0.85    # Aumentado de 0.75 para 0.85 (restritivo)

        # Log adicional para debug
        logging.info(f"🔍 Similaridades calculadas:")
        for i, (sim, (driver_id, driver_name)) in enumerate(zip(sims, id_name_pairs)):
            logging.info(f"   {driver_name}: {sim:.6f}")  # 6 casas decimais para mais precisão
        
        logging.info(f"🎯 Melhor match: {best_driver_name} com {best_sim:.6f}")
        logging.info(f"📏 Limiares: Strict={TH_STRICT}, Loose={TH_LOOSE}")

        authorized = best_sim >= TH_LOOSE
        if best_sim >= TH_STRICT:
            message = f"Motorista {best_driver_name} autorizado (alta confiança: {best_sim:.3f})"
        elif best_sim >= TH_LOOSE:
            message = f"Motorista {best_driver_name} autorizado (confiança moderada: {best_sim:.3f})"
        else:
            message = f"Motorista não reconhecido - similaridade: {best_sim:.6f} (necessário: {TH_LOOSE:.3f})"

        response_data = jsonify({
            "authorized": authorized,
            "message": message,
            "driver_id": best_driver_id if authorized else None,
            "driver_name": best_driver_name if authorized else "Desconhecido",
            "confidence": round(best_sim * 100, 1),
            "processing_time": round(time.perf_counter() - start, 2),
        }), 200

        logging.info("✅ Verificação concluída com sucesso")
        
        logging.info(f"🔍 Similaridades calculadas:")
        for i, (sim, (driver_id, driver_name)) in enumerate(zip(sims, id_name_pairs)):
            logging.info(f"   {driver_name}: {sim:.3f}")
        
        logging.info(f"🎯 Melhor match: {best_driver_name} com {best_sim:.3f}")
        logging.info(f"📏 Limiares: Strict={TH_STRICT}, Loose={TH_LOOSE}")

        return response_data
    except Exception as e:
        logging.error(f"❌ Erro na verificação: {str(e)}")
        logging.error(f"📋 Stack trace: {traceback.format_exc()}")
        return jsonify({"error": str(e), "authorized": False}), 500


# Adicione também logs na inicialização:
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    allowed_origins = os.environ.get("BACKEND_ALLOWED_ORIGINS", "")
    
    logging.info(f"🚀 Iniciando servidor Flask na porta {port}...")
    logging.info(f"🔒 CORS permitindo origens: {allowed_origins}")
    logging.info(f"📁 Diretório de trabalho: {os.getcwd()}")
    logging.info(f"🌍 Variáveis de ambiente relevantes:")
    logging.info(f"   - PORT: {port}")
    logging.info(f"   - BACKEND_ALLOWED_ORIGINS: {allowed_origins}")
    
    app.run(debug=True, host="0.0.0.0", port=port)