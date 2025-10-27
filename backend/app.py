import os
import io
import time
import base64
import logging
import traceback
from typing import List, Optional, Tuple, Dict
from collections import defaultdict
import statistics

import cv2
import numpy as np
import requests
from PIL import Image
from cachetools import TTLCache
from flask import Flask, request, jsonify
from flask_cors import CORS
from numpy.linalg import norm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# CORS via variável de ambiente
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("BACKEND_ALLOWED_ORIGINS", "").split(",") if o.strip()]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["*"]

CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}}, supports_credentials=True)

# Cache de embeddings: 10min, até 512 itens
driver_embedding_cache: TTLCache[str, List[np.ndarray]] = TTLCache(maxsize=512, ttl=600)

# Cache para histórico de similaridades (para threshold dinâmico)
similarity_history: List[float] = []
MAX_SIMILARITY_HISTORY = 1000

# Classificador Haar para detecção facial
_face_cascade: Optional[cv2.CascadeClassifier] = None


def get_face_cascade():
    """Obtém o classificador Haar para detecção facial"""
    global _face_cascade
    if _face_cascade is None:
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            _face_cascade = cv2.CascadeClassifier(cascade_path)
            logging.info("✅ Classificador Haar carregado")
        except Exception as e:
            logging.error(f"❌ Erro ao carregar classificador Haar: {e}")
            _face_cascade = None
    return _face_cascade


def detect_faces_opencv(rgb_image: np.ndarray) -> List[np.ndarray]:
    """Detecta rostos usando OpenCV Haar Cascades"""
    try:
        face_cascade = get_face_cascade()
        if face_cascade is None:
            return []
        
        # Converte para escala de cinza
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        
        # Detecta rostos
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            logging.warning("🔍 Nenhum rosto detectado com OpenCV")
            return []
        
        detected_faces = []
        for i, (x, y, w, h) in enumerate(faces):
            # Adiciona margem
            margin = int(w * 0.2)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(rgb_image.shape[1], x + w + margin)
            y2 = min(rgb_image.shape[0], y + h + margin)
            
            face_crop = rgb_image[y1:y2, x1:x2]
            if face_crop.size > 0:
                detected_faces.append(face_crop)
                logging.info(f"✅ Rosto {i} detectado com OpenCV: {face_crop.shape}")
        
        return detected_faces
        
    except Exception as e:
        logging.error(f"❌ Erro na detecção OpenCV: {e}")
        return []


def assess_face_quality(face_rgb: np.ndarray) -> Dict[str, float]:
    """Avalia qualidade do rosto detectado"""
    face_gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    
    # Brilho (0-1, ótimo ~0.5)
    brightness = np.mean(face_gray) / 255.0
    brightness_score = 1.0 - abs(brightness - 0.5) * 2
    
    # Contraste (0-1, maior é melhor)
    contrast = np.std(face_gray) / 128.0
    contrast_score = min(contrast, 1.0)
    
    # Nitidez via Laplaciano (0-1, maior é melhor)
    laplacian_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()
    sharpness_score = min(laplacian_var / 1000.0, 1.0)
    
    # Resolução (mínimo 80x80)
    resolution_score = min(min(face_rgb.shape[:2]) / 80.0, 1.0)
    
    # Score geral
    overall_score = (brightness_score + contrast_score + sharpness_score + resolution_score) / 4
    
    return {
        "overall": overall_score,
        "brightness": brightness_score,
        "contrast": contrast_score,
        "sharpness": sharpness_score,
        "resolution": resolution_score
    }


def extract_embedding_simple(rgb_image: np.ndarray) -> Optional[np.ndarray]:
    """Extrai embedding simples usando características OpenCV"""
    try:
        # Detecta rostos
        faces = detect_faces_opencv(rgb_image)
        
        if not faces:
            return None
        
        # Pega o maior rosto
        main_face = max(faces, key=lambda x: x.shape[0] * x.shape[1])
        
        # Avalia qualidade
        quality = assess_face_quality(main_face)
        logging.info(f"🎯 Qualidade da face: {quality['overall']:.3f}")
        
        if quality['overall'] < 0.3:
            logging.warning("⚠️ Qualidade de imagem muito baixa")
            return None
        
        # Normaliza o tamanho para 128x128
        face_resized = cv2.resize(main_face, (128, 128))
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)
        
        # Extrai características visuais simples mas efetivas
        features = []
        
        # 1. Histograma de intensidades (32 bins)
        hist = cv2.calcHist([face_gray], [0], None, [32], [0, 256])
        features.extend(hist.flatten())
        
        # 2. Divide em grid 8x8 e calcula estatísticas por região
        grid_size = 8
        cell_h = 128 // grid_size
        cell_w = 128 // grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                y_start = i * cell_h
                y_end = (i + 1) * cell_h
                x_start = j * cell_w
                x_end = (j + 1) * cell_w
                
                cell = face_gray[y_start:y_end, x_start:x_end]
                
                # Estatísticas básicas da célula
                features.extend([
                    np.mean(cell),           # Média
                    np.std(cell),            # Desvio padrão
                    np.median(cell),         # Mediana
                    np.min(cell),            # Mínimo
                    np.max(cell),            # Máximo
                ])
        
        # 3. Características de bordas (Sobel)
        grad_x = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Estatísticas dos gradientes
        features.extend([
            np.mean(np.abs(grad_x)),     # Bordas horizontais
            np.mean(np.abs(grad_y)),     # Bordas verticais
            np.mean(np.sqrt(grad_x**2 + grad_y**2)),  # Magnitude do gradiente
        ])
        
        # 4. Características de textura (LBP simplificado)
        # Divide em quadrantes e calcula variância
        h, w = face_gray.shape
        quadrants = [
            face_gray[0:h//2, 0:w//2],       # Superior esquerdo
            face_gray[0:h//2, w//2:w],       # Superior direito
            face_gray[h//2:h, 0:w//2],       # Inferior esquerdo
            face_gray[h//2:h, w//2:w],       # Inferior direito
        ]
        
        for quad in quadrants:
            features.extend([
                np.mean(quad),
                np.var(quad),
            ])
        
        # 5. Características de simetria (compara esquerda vs direita)
        left_half = face_gray[:, :w//2]
        right_half = np.fliplr(face_gray[:, w//2:])
        
        # Redimensiona para mesmo tamanho se necessário
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        symmetry_diff = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))
        features.append(symmetry_diff)
        
        # Converte para numpy array
        embedding = np.array(features, dtype=np.float32)
        
        # Padding/truncate para tamanho fixo (512 dimensões)
        target_size = 512
        if len(embedding) < target_size:
            # Repete características com pequeno ruído
            repeats = target_size // len(embedding) + 1
            extended = np.tile(embedding, repeats)
            # Adiciona ruído gaussiano pequeno para evitar repetições exatas
            noise = np.random.normal(0, 0.001, target_size)
            embedding = extended[:target_size] + noise
        else:
            embedding = embedding[:target_size]
        
        # Normalização
        embedding = embedding.astype(np.float32)
        
        # Normalização L2
        norm_val = np.linalg.norm(embedding)
        if norm_val > 1e-8:
            embedding = embedding / norm_val
        
        logging.info(f"✅ Embedding extraído: {embedding.shape}, norm: {np.linalg.norm(embedding):.6f}")
        return embedding
        
    except Exception as e:
        logging.error(f"❌ Erro na extração de embedding: {e}")
        return None


def calculate_dynamic_thresholds() -> Tuple[float, float]:
    """Calcula thresholds dinâmicos baseados no histórico"""
    if len(similarity_history) < 10:
        # Valores padrão ajustados para embeddings simples
        return 0.75, 0.65  # strict, loose
    
    # Calcula estatísticas do histórico
    mean_sim = statistics.mean(similarity_history)
    std_sim = statistics.stdev(similarity_history) if len(similarity_history) > 1 else 0.1
    
    # Thresholds baseados em desvios padrão
    strict_threshold = min(0.9, max(0.7, mean_sim + std_sim))
    loose_threshold = min(0.8, max(0.5, mean_sim))
    
    logging.info(f"📊 Thresholds dinâmicos: Strict={strict_threshold:.3f}, "
                f"Loose={loose_threshold:.3f} (baseado em {len(similarity_history)} amostras)")
    
    return strict_threshold, loose_threshold


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calcula similaridade cosseno entre dois embeddings"""
    return float(np.dot(v1, v2) / (norm(v1) * norm(v2) + 1e-12))


def get_average_embedding(embeddings: List[np.ndarray]) -> np.ndarray:
    """Calcula embedding médio de múltiplas fotos"""
    if len(embeddings) == 1:
        return embeddings[0]
    
    # Média simples
    avg_embedding = np.mean(embeddings, axis=0)
    
    # Renormaliza
    norm_val = np.linalg.norm(avg_embedding)
    if norm_val > 1e-8:
        avg_embedding = avg_embedding / norm_val
    
    logging.info(f"📊 Embedding médio calculado de {len(embeddings)} fotos")
    return avg_embedding.astype(np.float32)


def b64_to_rgb(image_data_uri: str) -> np.ndarray:
    """Converte imagem base64 para array RGB"""
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


def get_driver_embeddings(drivers: List[dict]) -> Tuple[List[np.ndarray], List[Tuple[str, str]]]:
    """Processa embeddings dos motoristas autorizados"""
    final_embeddings: List[np.ndarray] = []
    id_name_pairs: List[Tuple[str, str]] = []

    # Limpa cache de IDs que não estão na lista atual
    incoming_ids = {d.get("id") for d in drivers if d.get("id")}
    for driver_id in list(driver_embedding_cache.keys()):
        if driver_id not in incoming_ids:
            driver_embedding_cache.pop(driver_id, None)
            logging.info(f"🗑️ Removido do cache: {driver_id}")

    for driver_data in drivers:
        driver_id = driver_data.get("id")
        name = driver_data.get("name")
        photo_url = driver_data.get("photo_url")
        
        if not (driver_id and name and photo_url):
            logging.warning(f"⚠️ Dados incompletos do motorista: {driver_data}")
            continue

        # Verifica cache
        cached_embeddings = driver_embedding_cache.get(driver_id)
        
        if cached_embeddings is None:
            try:
                logging.info(f"📥 Baixando foto de {name} ({driver_id})...")
                response = requests.get(photo_url, timeout=15)
                response.raise_for_status()
                
                img = Image.open(io.BytesIO(response.content)).convert("RGB")
                embedding = extract_embedding_simple(np.array(img))
                
                if embedding is not None:
                    driver_embedding_cache[driver_id] = [embedding]
                    cached_embeddings = [embedding]
                    logging.info(f"✅ Embedding cacheado para {name} ({driver_id})")
                else:
                    logging.warning(f"❌ Falha ao extrair embedding de {name} ({driver_id})")
                    continue
                    
            except Exception as e:
                logging.error(f"❌ Erro ao processar {name} ({driver_id}): {e}")
                continue

        # Calcula embedding médio se há múltiplas fotos
        if len(cached_embeddings) > 1:
            final_embedding = get_average_embedding(cached_embeddings)
        else:
            final_embedding = cached_embeddings[0]

        final_embeddings.append(final_embedding)
        id_name_pairs.append((driver_id, name))

    return final_embeddings, id_name_pairs


@app.route("/health", methods=["GET"])
def health():
    """Endpoint de saúde da API"""
    try:
        face_cascade = get_face_cascade()
        detection_status = "✅ OpenCV Haar Cascades" if face_cascade else "❌ Detecção não disponível"
        
        return jsonify({
            "status": "ok",
            "message": f"API de reconhecimento facial online! {detection_status}",
            "face_detection": "OpenCV Haar Cascades",
            "embedding_method": "Features visuais customizadas",
            "cache_size": len(driver_embedding_cache),
            "similarity_history": len(similarity_history)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Erro na API: {str(e)}"
        }), 500


@app.route("/verify_driver", methods=["POST"])
def verify_driver():
    """Endpoint principal de verificação de motorista"""
    logging.info("🔄 Requisição recebida em /verify_driver")
    
    try:
        start_time = time.perf_counter()
        payload = request.get_json(silent=True) or {}

        # Validação de entrada
        image_data = payload.get("image")
        authorized_drivers = payload.get("authorized_drivers", [])
        car_id = payload.get("car_id", "N/A")

        if not image_data or not isinstance(authorized_drivers, list):
            return jsonify({
                "error": "Requisição inválida. Envie 'image' (base64) e 'authorized_drivers' (lista)."
            }), 400

        logging.info(f"🚗 Verificação do carro {car_id} com {len(authorized_drivers)} motoristas")

        # Converte imagem base64 para RGB
        try:
            rgb_image = b64_to_rgb(image_data)
            logging.info(f"🖼️ Imagem processada: {rgb_image.shape}")
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        # Extrai embedding da captura
        query_embedding = extract_embedding_simple(rgb_image)
        if query_embedding is None:
            return jsonify({
                "authorized": False,
                "message": "❌ Nenhum rosto detectado na captura ou qualidade insuficiente.",
                "driver_id": None,
                "driver_name": "Desconhecido",
                "confidence": 0,
                "processing_time": round(time.perf_counter() - start_time, 2),
                "quality_recommendation": "Melhore a iluminação e posicione o rosto de frente para a câmera"
            }), 200

        # Processa embeddings dos motoristas autorizados
        known_embeddings, id_name_pairs = get_driver_embeddings(authorized_drivers)
        if not known_embeddings:
            return jsonify({
                "authorized": False,
                "message": "❌ Nenhum motorista autorizado válido encontrado.",
                "driver_id": None,
                "driver_name": "Desconhecido",
                "confidence": 0,
                "processing_time": round(time.perf_counter() - start_time, 2),
            }), 200

        # Calcula similaridades
        similarities = [cosine_similarity(query_embedding, emb) for emb in known_embeddings]
        best_idx = int(np.argmax(similarities))
        best_similarity = float(similarities[best_idx])
        best_driver_id, best_driver_name = id_name_pairs[best_idx]

        # Adiciona ao histórico para threshold dinâmico
        similarity_history.append(best_similarity)
        if len(similarity_history) > MAX_SIMILARITY_HISTORY:
            similarity_history.pop(0)

        # Calcula thresholds dinâmicos
        th_strict, th_loose = calculate_dynamic_thresholds()

        # Logs detalhados
        logging.info(f"🔍 Similaridades calculadas:")
        for sim, (driver_id, driver_name) in zip(similarities, id_name_pairs):
            logging.info(f"   {driver_name}: {sim:.6f}")
        
        logging.info(f"🎯 Melhor match: {best_driver_name} com {best_similarity:.6f}")
        logging.info(f"📏 Thresholds: Strict={th_strict:.3f}, Loose={th_loose:.3f}")

        # Determina autorização
        authorized = best_similarity >= th_loose
        confidence_level = "alta" if best_similarity >= th_strict else "moderada" if authorized else "baixa"
        
        if authorized:
            message = f"✅ Motorista {best_driver_name} autorizado ({confidence_level} confiança: {best_similarity:.3f})"
        else:
            message = f"❌ Motorista não reconhecido - similaridade: {best_similarity:.6f} (necessário: {th_loose:.3f})"

        # Recomendações baseadas na similaridade
        recommendations = []
        if not authorized:
            if best_similarity > 0.4:
                recommendations.append("Tente melhorar a iluminação")
                recommendations.append("Posicione o rosto de frente para a câmera")
            else:
                recommendations.append("Verifique se este motorista está cadastrado")

        response_data = {
            "authorized": authorized,
            "message": message,
            "driver_id": best_driver_id if authorized else None,
            "driver_name": best_driver_name if authorized else "Desconhecido",
            "confidence": round(best_similarity * 100, 1),
            "confidence_level": confidence_level,
            "processing_time": round(time.perf_counter() - start_time, 2),
            "thresholds": {"strict": th_strict, "loose": th_loose},
            "recommendations": recommendations,
            "all_similarities": [
                {"name": name, "similarity": round(sim, 6)} 
                for sim, (_, name) in zip(similarities, id_name_pairs)
            ]
        }

        logging.info("✅ Verificação concluída com sucesso")
        return jsonify(response_data), 200

    except Exception as e:
        logging.error(f"❌ Erro na verificação: {str(e)}")
        logging.error(f"📋 Stack trace: {traceback.format_exc()}")
        return jsonify({
            "error": f"Erro interno: {str(e)}",
            "authorized": False
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    
    logging.info(f"🚀 Iniciando servidor Flask na porta {port}...")
    logging.info(f"🔍 Detecção facial: OpenCV Haar Cascades")
    logging.info(f"🧠 Método de embedding: Características visuais customizadas")
    
    # Testa se o classificador Haar funciona
    try:
        cascade = get_face_cascade()
        if cascade:
            logging.info("✅ Sistema de reconhecimento facial pronto!")
        else:
            logging.error("❌ Falha ao carregar classificador de faces")
    except Exception as e:
        logging.error(f"❌ Erro no sistema de detecção: {e}")
    
    app.run(debug=True, host="0.0.0.0", port=port)

    #SADAD