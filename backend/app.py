"""
AutoGuard Vision Backend - Sistema de Reconhecimento Facial
Versão otimizada com melhor estrutura, tratamento de erros e performance
"""

import os
import io
import time
import base64
import logging
import traceback
import functools
from typing import List, Optional, Tuple, Dict, Any
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import statistics

import cv2
import numpy as np
import requests
from PIL import Image
from cachetools import TTLCache
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from numpy.linalg import norm

# =====================================================
# CONFIGURAÇÃO E CONSTANTES
# =====================================================

# Configuração de logging mais robusta
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('facial_recognition.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Constantes do sistema
class Config:
    MIN_FACE_AREA = 5000
    MIN_QUALITY_SCORE = 0.4
    MIN_CONFIDENCE_THRESHOLD = 0.7
    CACHE_TTL = 600  # 10 minutos
    CACHE_MAX_SIZE = 512
    MAX_SIMILARITY_HISTORY = 1000
    REQUEST_TIMEOUT = 15
    EMBEDDING_DIMENSIONS = 1024
    FACE_SIZE = 128

# Enums para melhor organização
class ValidationReason(Enum):
    VALID_FACE = "valid_face"
    FACE_TOO_SMALL = "face_too_small"
    INVALID_PROPORTIONS = "invalid_proportions"
    NO_EYES_DETECTED = "no_eyes_detected"
    TOO_UNIFORM = "too_uniform"
    LOW_GRADIENTS = "low_gradients"
    VALIDATION_ERROR = "validation_error"

@dataclass
class ValidationResult:
    is_valid: bool
    reason: ValidationReason
    details: Any
    confidence: float = 0.0

@dataclass
class QualityMetrics:
    overall: float
    brightness: float
    contrast: float
    sharpness: float
    resolution: float
    uniformity: float

# =====================================================
# INICIALIZAÇÃO DA APLICAÇÃO
# =====================================================

app = Flask(__name__)

# Configuração CORS mais segura
def setup_cors():
    allowed_origins = [
        origin.strip() 
        for origin in os.environ.get("BACKEND_ALLOWED_ORIGINS", "").split(",") 
        if origin.strip()
    ]
    if not allowed_origins:
        allowed_origins = ["*"]  # Apenas para desenvolvimento
    
    CORS(app, resources={
        r"/*": {
            "origins": allowed_origins,
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    }, supports_credentials=True)

setup_cors()

# Caches globais
driver_embedding_cache: TTLCache[str, List[np.ndarray]] = TTLCache(
    maxsize=Config.CACHE_MAX_SIZE, 
    ttl=Config.CACHE_TTL
)
similarity_history: List[float] = []

# Classificadores globais
_face_cascade: Optional[cv2.CascadeClassifier] = None
_eye_cascade: Optional[cv2.CascadeClassifier] = None

# =====================================================
# UTILITÁRIOS E HELPERS
# =====================================================

def log_request_info():
    """Log informações da requisição atual"""
    g.start_time = time.perf_counter()
    logger.info(f"📥 {request.method} {request.path} - IP: {request.remote_addr}")

def log_response_info(response_data: dict):
    """Log informações da resposta"""
    if hasattr(g, 'start_time'):
        duration = time.perf_counter() - g.start_time
        logger.info(f"📤 Resposta em {duration:.3f}s - Status: {response_data.get('status', 'unknown')}")

def create_error_response(message: str, status_code: int = 500, details: Any = None) -> tuple:
    """Cria resposta de erro padronizada"""
    error_response = {
        "status": "error",
        "message": message,
        "authorized": False
    }
    if details:
        error_response["details"] = details
    
    log_response_info(error_response)
    return jsonify(error_response), status_code

def create_success_response(data: dict, message: str = "Operação realizada com sucesso") -> tuple:
    """Cria resposta de sucesso padronizada"""
    success_response = {
        "status": "success",
        "message": message,
        **data
    }
    log_response_info(success_response)
    return jsonify(success_response), 200

# =====================================================
# INICIALIZAÇÃO DE COMPONENTES
# =====================================================

@functools.lru_cache(maxsize=1)
def get_face_cascade() -> Optional[cv2.CascadeClassifier]:
    """Obtém o classificador Haar para detecção facial com cache"""
    global _face_cascade
    if _face_cascade is None:
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            _face_cascade = cv2.CascadeClassifier(cascade_path)
            if _face_cascade.empty():
                raise Exception("Classificador não foi carregado corretamente")
            logger.info("✅ Classificador Haar de faces carregado")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar classificador Haar de faces: {e}")
            _face_cascade = None
    return _face_cascade

@functools.lru_cache(maxsize=1)
def get_eye_cascade() -> Optional[cv2.CascadeClassifier]:
    """Obtém o classificador Haar para detecção de olhos com cache"""
    global _eye_cascade
    if _eye_cascade is None:
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            _eye_cascade = cv2.CascadeClassifier(cascade_path)
            if _eye_cascade.empty():
                raise Exception("Classificador de olhos não foi carregado corretamente")
            logger.info("✅ Classificador Haar de olhos carregado")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar classificador Haar de olhos: {e}")
            _eye_cascade = None
    return _eye_cascade

# =====================================================
# PROCESSAMENTO DE IMAGENS
# =====================================================

def validate_image_input(image_data: str) -> np.ndarray:
    """Valida e converte imagem base64 para array RGB"""
    try:
        if not image_data:
            raise ValueError("Dados de imagem não fornecidos")
        
        # Remove prefixo data URL se presente
        if "," in image_data:
            _, enc = image_data.split(",", 1)
        else:
            enc = image_data
        
        # Decodifica base64
        raw = base64.b64decode(enc)
        if len(raw) == 0:
            raise ValueError("Dados base64 vazios")
        
        # Converte para imagem PIL e depois para array numpy
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        
        # Valida dimensões mínimas
        if img.size[0] < 100 or img.size[1] < 100:
            raise ValueError(f"Imagem muito pequena: {img.size}. Mínimo: 100x100")
        
        return np.array(img)
        
    except Exception as e:
        logger.error(f"❌ Erro ao processar imagem base64: {e}")
        raise ValueError(f"Imagem base64 inválida: {str(e)}")

def validate_face_authenticity(face_rgb: np.ndarray) -> ValidationResult:
    """Valida se é realmente um rosto humano usando múltiplos critérios"""
    try:
        h, w = face_rgb.shape[:2]
        area = h * w
        
        # 1. Verifica área mínima
        if area < Config.MIN_FACE_AREA:
            return ValidationResult(
                is_valid=False,
                reason=ValidationReason.FACE_TOO_SMALL,
                details=f"Área da face: {area} < {Config.MIN_FACE_AREA}",
                confidence=0.1
            )
        
        # 2. Verifica proporções (faces humanas têm proporções típicas)
        aspect_ratio = w / h
        if not 0.6 <= aspect_ratio <= 1.4:
            return ValidationResult(
                is_valid=False,
                reason=ValidationReason.INVALID_PROPORTIONS,
                details=f"Proporção inválida: {aspect_ratio:.2f}",
                confidence=0.2
            )
        
        # 3. Detecta olhos na face
        eye_cascade = get_eye_cascade()
        eyes_detected = 0
        if eye_cascade is not None:
            face_gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
            eyes = eye_cascade.detectMultiScale(
                face_gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(20, 20)
            )
            eyes_detected = len(eyes)
            
            if eyes_detected < 1:
                return ValidationResult(
                    is_valid=False,
                    reason=ValidationReason.NO_EYES_DETECTED,
                    details=f"Olhos detectados: {eyes_detected}",
                    confidence=0.3
                )
        
        # 4. Verifica uniformidade
        face_gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
        std_dev = np.std(face_gray)
        if std_dev < 15:
            return ValidationResult(
                is_valid=False,
                reason=ValidationReason.TOO_UNIFORM,
                details=f"Desvio padrão muito baixo: {std_dev:.2f}",
                confidence=0.4
            )
        
        # 5. Verifica gradientes
        grad_x = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        avg_gradient = np.mean(gradient_magnitude)
        
        if avg_gradient < 10:
            return ValidationResult(
                is_valid=False,
                reason=ValidationReason.LOW_GRADIENTS,
                details=f"Gradiente médio muito baixo: {avg_gradient:.2f}",
                confidence=0.5
            )
        
        # Calcula confiança baseada nos critérios
        confidence = min(1.0, (
            (area / Config.MIN_FACE_AREA) * 0.2 +
            (1 - abs(aspect_ratio - 1.0)) * 0.2 +
            min(eyes_detected / 2, 1.0) * 0.2 +
            min(std_dev / 50, 1.0) * 0.2 +
            min(avg_gradient / 50, 1.0) * 0.2
        ))
        
        return ValidationResult(
            is_valid=True,
            reason=ValidationReason.VALID_FACE,
            details={
                "area": area,
                "aspect_ratio": aspect_ratio,
                "eyes_detected": eyes_detected,
                "std_dev": std_dev,
                "avg_gradient": avg_gradient
            },
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"❌ Erro na validação de autenticidade: {e}")
        return ValidationResult(
            is_valid=False,
            reason=ValidationReason.VALIDATION_ERROR,
            details=str(e),
            confidence=0.0
        )

def detect_and_validate_faces(rgb_image: np.ndarray) -> List[Tuple[np.ndarray, ValidationResult]]:
    """Detecta e valida rostos usando OpenCV Haar Cascades"""
    try:
        face_cascade = get_face_cascade()
        if face_cascade is None:
            logger.warning("🔍 Classificador de faces não disponível")
            return []
        
        # Converte para escala de cinza
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        
        # Detecta rostos com parâmetros otimizados
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(80, 80),
            maxSize=(400, 400),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            logger.warning("🔍 Nenhum rosto detectado com OpenCV")
            return []
        
        validated_faces = []
        for i, (x, y, w, h) in enumerate(faces):
            # Adiciona margem proporcional
            margin = int(w * 0.2)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(rgb_image.shape[1], x + w + margin)
            y2 = min(rgb_image.shape[0], y + h + margin)
            
            face_crop = rgb_image[y1:y2, x1:x2]
            if face_crop.size > 0:
                validation = validate_face_authenticity(face_crop)
                
                if validation.is_valid:
                    validated_faces.append((face_crop, validation))
                    logger.info(f"✅ Rosto {i} validado: {face_crop.shape} - confiança: {validation.confidence:.3f}")
                else:
                    logger.warning(f"❌ Rosto {i} rejeitado: {validation.reason.value} - {validation.details}")
        
        return validated_faces
        
    except Exception as e:
        logger.error(f"❌ Erro na detecção de faces: {e}")
        return []

def assess_face_quality(face_rgb: np.ndarray) -> QualityMetrics:
    """Avalia qualidade do rosto detectado com critérios aprimorados"""
    try:
        face_gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
        
        # Brilho (0-1, ótimo ~0.5)
        brightness = np.mean(face_gray) / 255.0
        brightness_score = max(0, 1.0 - abs(brightness - 0.5) * 2)
        
        # Contraste (0-1, maior é melhor)
        contrast = np.std(face_gray) / 128.0
        contrast_score = min(contrast, 1.0)
        
        # Nitidez via Laplaciano (0-1, maior é melhor)
        laplacian_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 1000.0, 1.0)
        
        # Resolução (mínimo 100x100)
        resolution_score = min(min(face_rgb.shape[:2]) / 100.0, 1.0)
        
        # Uniformidade (penaliza imagens muito uniformes)
        uniformity = 1.0 - min(np.std(face_gray) / 64.0, 1.0)
        uniformity_score = max(0, 1.0 - uniformity)
        
        # Score geral ponderado
        weights = [0.2, 0.25, 0.25, 0.15, 0.15]
        scores = [brightness_score, contrast_score, sharpness_score, resolution_score, uniformity_score]
        overall_score = sum(w * s for w, s in zip(weights, scores))
        
        return QualityMetrics(
            overall=overall_score,
            brightness=brightness_score,
            contrast=contrast_score,
            sharpness=sharpness_score,
            resolution=resolution_score,
            uniformity=uniformity_score
        )
        
    except Exception as e:
        logger.error(f"❌ Erro na avaliação de qualidade: {e}")
        return QualityMetrics(0, 0, 0, 0, 0, 0)

# =====================================================
# EXTRAÇÃO DE EMBEDDINGS
# =====================================================

class EmbeddingExtractor:
    """Classe responsável pela extração de embeddings faciais"""
    
    @staticmethod
    def extract_visual_features(face_gray: np.ndarray) -> np.ndarray:
        """Extrai características visuais avançadas da face"""
        features = []
        
        # 1. Histograma de intensidades
        hist = cv2.calcHist([face_gray], [0], None, [64], [0, 256])
        features.extend(hist.flatten())
        
        # 2. Análise por grid 16x16
        grid_size = 16
        cell_h = Config.FACE_SIZE // grid_size
        cell_w = Config.FACE_SIZE // grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                y_start = i * cell_h
                y_end = (i + 1) * cell_h
                x_start = j * cell_w
                x_end = (j + 1) * cell_w
                
                cell = face_gray[y_start:y_end, x_start:x_end]
                
                # Estatísticas da célula
                features.extend([
                    np.mean(cell),
                    np.std(cell),
                    np.median(cell),
                    np.min(cell),
                    np.max(cell),
                    np.percentile(cell, 25),
                    np.percentile(cell, 75),
                ])
        
        # 3. Gradientes Sobel multi-escala
        for ksize in [3, 5]:
            grad_x = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=ksize)
            grad_y = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=ksize)
            
            features.extend([
                np.mean(np.abs(grad_x)),
                np.mean(np.abs(grad_y)),
                np.mean(np.sqrt(grad_x**2 + grad_y**2)),
                np.std(grad_x),
                np.std(grad_y),
            ])
        
        # 4. LBP (Local Binary Pattern)
        lbp = EmbeddingExtractor._calculate_lbp(face_gray)
        lbp_hist = np.histogram(lbp, bins=32, range=(0, 255))[0]
        features.extend(lbp_hist)
        
        # 5. Características de simetria
        h, w = face_gray.shape
        left_half = face_gray[:, :w//2]
        right_half = np.fliplr(face_gray[:, w//2:])
        
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        symmetry_diff = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))
        try:
            symmetry_corr = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
            if np.isnan(symmetry_corr):
                symmetry_corr = 0.0
        except:
            symmetry_corr = 0.0
        
        features.extend([symmetry_diff, symmetry_corr])
        
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def _calculate_lbp(image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """Calcula Local Binary Pattern"""
        lbp = np.zeros_like(image)
        for i in range(radius, image.shape[0] - radius):
            for j in range(radius, image.shape[1] - radius):
                center = image[i, j]
                binary_string = ''
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    if image[x, y] >= center:
                        binary_string += '1'
                    else:
                        binary_string += '0'
                lbp[i, j] = int(binary_string, 2)
        return lbp
    
    @staticmethod
    def normalize_embedding(embedding: np.ndarray) -> Optional[np.ndarray]:
        """Normaliza embedding para tamanho fixo com L2 normalization"""
        try:
            # Padding/truncate para tamanho fixo
            if len(embedding) < Config.EMBEDDING_DIMENSIONS:
                repeats = Config.EMBEDDING_DIMENSIONS // len(embedding) + 1
                extended = np.tile(embedding, repeats)
                # Adiciona ruído pequeno para evitar repetições exatas
                noise = np.random.normal(0, 0.0001, Config.EMBEDDING_DIMENSIONS)
                embedding = extended[:Config.EMBEDDING_DIMENSIONS] + noise
            else:
                embedding = embedding[:Config.EMBEDDING_DIMENSIONS]
            
            # Normalização L2
            embedding = embedding.astype(np.float32)
            norm_val = np.linalg.norm(embedding)
            
            if norm_val > 1e-12:
                embedding = embedding / norm_val
            else:
                logger.error("❌ Embedding com norma zero!")
                return None
            
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Erro na normalização do embedding: {e}")
            return None

def extract_embedding_from_image(rgb_image: np.ndarray) -> Optional[Dict[str, Any]]:
    """Extrai embedding principal de uma imagem com informações detalhadas"""
    try:
        # Detecta e valida faces
        validated_faces = detect_and_validate_faces(rgb_image)
        
        if not validated_faces:
            logger.warning("❌ Nenhum rosto válido detectado")
            return None
        
        # Seleciona o melhor rosto (maior área e melhor validação)
        best_face, best_validation = max(
            validated_faces, 
            key=lambda x: x[0].shape[0] * x[0].shape[1] * x[1].confidence
        )
        
        # Avalia qualidade
        quality = assess_face_quality(best_face)
        
        logger.info(f"🎯 Qualidade da face: {quality.overall:.3f} "
                   f"(brilho: {quality.brightness:.2f}, "
                   f"contraste: {quality.contrast:.2f}, "
                   f"nitidez: {quality.sharpness:.2f})")
        
        # Verifica qualidade mínima
        if quality.overall < Config.MIN_QUALITY_SCORE:
            logger.warning(f"⚠️ Qualidade insuficiente: {quality.overall:.3f} < {Config.MIN_QUALITY_SCORE}")
            return None
        
        # Preprocessa face
        face_resized = cv2.resize(best_face, (Config.FACE_SIZE, Config.FACE_SIZE))
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)
        
        # Extrai características
        features = EmbeddingExtractor.extract_visual_features(face_gray)
        
        # Normaliza embedding
        embedding = EmbeddingExtractor.normalize_embedding(features)
        
        if embedding is None:
            return None
        
        result = {
            "embedding": embedding,
            "quality": quality,
            "validation": best_validation,
            "face_shape": best_face.shape
        }
        
        logger.info(f"✅ Embedding extraído: {embedding.shape}, norm: {np.linalg.norm(embedding):.6f}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Erro na extração de embedding: {e}")
        return None

# =====================================================
# ALGORITMOS DE SIMILARIDADE E THRESHOLDS
# =====================================================

def calculate_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calcula similaridade cosseno entre dois embeddings"""
    try:
        dot_product = np.dot(v1, v2)
        norm_product = norm(v1) * norm(v2)
        if norm_product < 1e-12:
            return 0.0
        return float(dot_product / norm_product)
    except Exception as e:
        logger.error(f"❌ Erro no cálculo de similaridade: {e}")
        return 0.0

def calculate_dynamic_thresholds() -> Tuple[float, float]:
    """Calcula thresholds dinâmicos baseados no histórico de similaridades"""
    if len(similarity_history) < 20:
        return 0.85, 0.75  # strict, loose - valores conservadores
    
    try:
        mean_sim = statistics.mean(similarity_history)
        std_sim = statistics.stdev(similarity_history) if len(similarity_history) > 1 else 0.05
        
        # Thresholds adaptativos mais conservadores
        strict_threshold = min(0.95, max(0.8, mean_sim + 1.5 * std_sim))
        loose_threshold = min(0.9, max(0.7, mean_sim + 0.5 * std_sim))
        
        # Garante thresholds mínimos
        strict_threshold = max(strict_threshold, Config.MIN_CONFIDENCE_THRESHOLD + 0.1)
        loose_threshold = max(loose_threshold, Config.MIN_CONFIDENCE_THRESHOLD)
        
        logger.info(f"📊 Thresholds dinâmicos: Strict={strict_threshold:.3f}, "
                   f"Loose={loose_threshold:.3f} (baseado em {len(similarity_history)} amostras)")
        
        return strict_threshold, loose_threshold
        
    except Exception as e:
        logger.error(f"❌ Erro no cálculo de thresholds: {e}")
        return 0.85, 0.75

def calculate_weighted_average_embedding(embeddings: List[np.ndarray]) -> np.ndarray:
    """Calcula embedding médio ponderado de múltiplas fotos"""
    if len(embeddings) == 1:
        return embeddings[0]
    
    try:
        # Pesos exponenciais (mais recentes têm maior peso)
        weights = np.exp(np.linspace(-1, 0, len(embeddings)))
        weights = weights / np.sum(weights)
        
        weighted_sum = np.zeros_like(embeddings[0])
        for emb, weight in zip(embeddings, weights):
            weighted_sum += weight * emb
        
        # Renormaliza
        norm_val = np.linalg.norm(weighted_sum)
        if norm_val > 1e-12:
            weighted_sum = weighted_sum / norm_val
        
        logger.info(f"📊 Embedding médio ponderado calculado de {len(embeddings)} fotos")
        return weighted_sum.astype(np.float32)
        
    except Exception as e:
        logger.error(f"❌ Erro no cálculo do embedding médio: {e}")
        return embeddings[0]  # Fallback para o primeiro embedding

# =====================================================
# PROCESSAMENTO DE MOTORISTAS
# =====================================================

def process_driver_embeddings(drivers: List[dict]) -> Tuple[List[np.ndarray], List[Tuple[str, str]]]:
    """Processa embeddings dos motoristas autorizados com cache otimizado"""
    final_embeddings: List[np.ndarray] = []
    id_name_pairs: List[Tuple[str, str]] = []

    # Limpa cache de IDs não presentes
    incoming_ids = {d.get("id") for d in drivers if d.get("id")}
    for driver_id in list(driver_embedding_cache.keys()):
        if driver_id not in incoming_ids:
            driver_embedding_cache.pop(driver_id, None)
            logger.info(f"🗑️ Removido do cache: {driver_id}")

    for driver_data in drivers:
        driver_id = driver_data.get("id")
        name = driver_data.get("name")
        photo_url = driver_data.get("photo_url")
        
        if not all([driver_id, name, photo_url]):
            logger.warning(f"⚠️ Dados incompletos do motorista: {driver_data}")
            continue

        try:
            # Verifica cache
            cached_embeddings = driver_embedding_cache.get(driver_id)
            
            if cached_embeddings is None:
                logger.info(f"📥 Processando foto de {name} ({driver_id})...")
                
                # Download com timeout
                response = requests.get(photo_url, timeout=Config.REQUEST_TIMEOUT)
                response.raise_for_status()
                
                # Processa imagem
                img = Image.open(io.BytesIO(response.content)).convert("RGB")
                embedding_result = extract_embedding_from_image(np.array(img))
                
                if embedding_result and embedding_result["embedding"] is not None:
                    driver_embedding_cache[driver_id] = [embedding_result["embedding"]]
                    cached_embeddings = [embedding_result["embedding"]]
                    logger.info(f"✅ Embedding cacheado para {name} ({driver_id})")
                else:
                    logger.warning(f"❌ Falha ao extrair embedding de {name} ({driver_id})")
                    continue

            # Calcula embedding final (médio se múltiplas fotos)
            if len(cached_embeddings) > 1:
                final_embedding = calculate_weighted_average_embedding(cached_embeddings)
            else:
                final_embedding = cached_embeddings[0]

            final_embeddings.append(final_embedding)
            id_name_pairs.append((driver_id, name))
            
        except requests.RequestException as e:
            logger.error(f"❌ Erro de rede ao processar {name} ({driver_id}): {e}")
            continue
        except Exception as e:
            logger.error(f"❌ Erro ao processar {name} ({driver_id}): {e}")
            continue

    return final_embeddings, id_name_pairs

# =====================================================
# ROTAS DA API
# =====================================================

@app.before_request
def before_request():
    """Hook executado antes de cada requisição"""
    log_request_info()

@app.route("/health", methods=["GET"])
def health_check():
    """Endpoint de verificação de saúde da API"""
    try:
        face_cascade = get_face_cascade()
        eye_cascade = get_eye_cascade()
        
        health_data = {
            "face_detection": "✅ OpenCV Haar Cascades" if face_cascade else "❌ Não disponível",
            "eye_detection": "✅ Detecção de olhos" if eye_cascade else "❌ Não disponível",
            "embedding_method": "Features visuais aprimoradas",
            "embedding_dimensions": Config.EMBEDDING_DIMENSIONS,
            "cache_size": len(driver_embedding_cache),
            "similarity_history_size": len(similarity_history),
            "system_config": {
                "min_confidence": Config.MIN_CONFIDENCE_THRESHOLD,
                "min_quality": Config.MIN_QUALITY_SCORE,
                "min_face_area": Config.MIN_FACE_AREA
            }
        }
        
        return create_success_response(
            health_data, 
            "API de reconhecimento facial online!"
        )
        
    except Exception as e:
        logger.error(f"❌ Erro no health check: {e}")
        return create_error_response(f"Erro no sistema: {str(e)}")

@app.route("/verify_driver", methods=["POST"])
def verify_driver():
    """Endpoint principal de verificação de motorista"""
    try:
        # Validação da requisição
        payload = request.get_json(silent=True)
        if not payload:
            return create_error_response("Payload JSON inválido", 400)

        image_data = payload.get("image")
        authorized_drivers = payload.get("authorized_drivers", [])
        car_id = payload.get("car_id", "N/A")

        if not image_data or not isinstance(authorized_drivers, list):
            return create_error_response(
                "Requisição inválida. Envie 'image' (base64) e 'authorized_drivers' (lista).", 
                400
            )

        logger.info(f"🚗 Verificação do carro {car_id} com {len(authorized_drivers)} motoristas")

        # Processamento da imagem
        try:
            rgb_image = validate_image_input(image_data)
            logger.info(f"🖼️ Imagem processada: {rgb_image.shape}")
        except ValueError as e:
            return create_error_response(str(e), 400)

        # Extração de embedding da captura
        embedding_result = extract_embedding_from_image(rgb_image)
        if not embedding_result or embedding_result["embedding"] is None:
            return create_success_response({
                "authorized": False,
                "driver_id": None,
                "driver_name": "Desconhecido",
                "confidence": 0,
                "confidence_level": "baixa",
                "quality_recommendation": "Posicione o rosto de frente para a câmera com boa iluminação. Certifique-se de que ambos os olhos estão visíveis."
            }, "Nenhum rosto válido detectado na captura")

        query_embedding = embedding_result["embedding"]

        # Processamento dos motoristas autorizados
        known_embeddings, id_name_pairs = process_driver_embeddings(authorized_drivers)
        if not known_embeddings:
            return create_success_response({
                "authorized": False,
                "driver_id": None,
                "driver_name": "Desconhecido",
                "confidence": 0,
                "confidence_level": "baixa"
            }, "Nenhum motorista autorizado válido encontrado")

        # Cálculo de similaridades
        similarities = [
            calculate_cosine_similarity(query_embedding, emb) 
            for emb in known_embeddings
        ]
        
        best_idx = int(np.argmax(similarities))
        best_similarity = float(similarities[best_idx])
        best_driver_id, best_driver_name = id_name_pairs[best_idx]

        # Atualização do histórico
        similarity_history.append(best_similarity)
        if len(similarity_history) > Config.MAX_SIMILARITY_HISTORY:
            similarity_history.pop(0)

        # Cálculo de thresholds dinâmicos
        th_strict, th_loose = calculate_dynamic_thresholds()

        # Logs detalhados
        logger.info("🔍 Similaridades calculadas:")
        for sim, (driver_id, driver_name) in zip(similarities, id_name_pairs):
            logger.info(f"   {driver_name}: {sim:.6f}")
        
        logger.info(f"🎯 Melhor match: {best_driver_name} com {best_similarity:.6f}")
        logger.info(f"📏 Thresholds: Strict={th_strict:.3f}, Loose={th_loose:.3f}")

        # Determinação de autorização
        authorized = best_similarity >= th_loose
        confidence_level = (
            "alta" if best_similarity >= th_strict else 
            "moderada" if authorized else 
            "baixa"
        )
        
        # Geração de recomendações
        recommendations = []
        if not authorized:
            if best_similarity > 0.5:
                recommendations.extend([
                    "Melhore a iluminação do ambiente",
                    "Posicione o rosto completamente de frente para a câmera",
                    "Remova óculos escuros ou acessórios que cubram o rosto",
                    "Certifique-se de que ambos os olhos estão claramente visíveis"
                ])
            elif best_similarity > 0.3:
                recommendations.extend([
                    "Verifique se você está cadastrado como motorista autorizado",
                    "Tente uma nova captura com melhor qualidade"
                ])
            else:
                recommendations.extend([
                    "Pessoa não reconhecida no sistema",
                    "Verifique se o cadastro foi realizado corretamente"
                ])

        # Resposta final
        verification_data = {
            "authorized": authorized,
            "driver_id": best_driver_id if authorized else None,
            "driver_name": best_driver_name if authorized else "Desconhecido",
            "confidence": round(best_similarity * 100, 1),
            "confidence_level": confidence_level,
            "thresholds": {"strict": th_strict, "loose": th_loose},
            "recommendations": recommendations,
            "all_similarities": [
                {"name": name, "similarity": round(sim, 6)} 
                for sim, (_, name) in zip(similarities, id_name_pairs)
            ],
            "system_info": {
                "min_confidence_threshold": Config.MIN_CONFIDENCE_THRESHOLD,
                "min_quality_score": Config.MIN_QUALITY_SCORE,
                "embedding_dimensions": len(query_embedding),
                "face_quality": {
                    "overall": round(embedding_result["quality"].overall, 3),
                    "brightness": round(embedding_result["quality"].brightness, 3),
                    "contrast": round(embedding_result["quality"].contrast, 3),
                    "sharpness": round(embedding_result["quality"].sharpness, 3)
                }
            }
        }

        message = (
            f"Motorista {best_driver_name} autorizado com {confidence_level} confiança" 
            if authorized else 
            f"Acesso negado. Similaridade insuficiente: {best_similarity:.3f} (necessário: ≥{th_loose:.3f})"
        )

        return create_success_response(verification_data, message)

    except Exception as e:
        logger.error(f"❌ Erro na verificação: {str(e)}")
        logger.error(f"📋 Stack trace: {traceback.format_exc()}")
        return create_error_response(f"Erro interno: {str(e)}")

@app.errorhandler(404)
def not_found(error):
    """Handler para rotas não encontradas"""
    return create_error_response("Endpoint não encontrado", 404)

@app.errorhandler(405)
def method_not_allowed(error):
    """Handler para métodos não permitidos"""
    return create_error_response("Método HTTP não permitido", 405)

@app.errorhandler(500)
def internal_error(error):
    """Handler para erros internos"""
    logger.error(f"Erro interno do servidor: {error}")
    return create_error_response("Erro interno do servidor", 500)

# =====================================================
# INICIALIZAÇÃO DA APLICAÇÃO
# =====================================================

def initialize_system():
    """Inicializa os componentes do sistema"""
    logger.info(f"🚀 Iniciando AutoGuard Vision Backend...")
    logger.info(f"🔍 Detecção facial: OpenCV Haar Cascades com validação aprimorada")
    logger.info(f"🧠 Método de embedding: Features visuais aprimoradas ({Config.EMBEDDING_DIMENSIONS}D)")
    logger.info(f"📊 Configurações: Min. Confiança={Config.MIN_CONFIDENCE_THRESHOLD}, Min. Qualidade={Config.MIN_QUALITY_SCORE}")
    
    # Testa componentes críticos
    try:
        face_cascade = get_face_cascade()
        eye_cascade = get_eye_cascade()
        
        if face_cascade and eye_cascade:
            logger.info("✅ Sistema de reconhecimento facial aprimorado pronto!")
        else:
            logger.warning("⚠️ Alguns classificadores não foram carregados - funcionalidade limitada")
            
    except Exception as e:
        logger.error(f"❌ Erro na inicialização dos componentes: {e}")

def main():
    """Função principal da aplicação"""
    initialize_system()
    
    port = int(os.environ.get("PORT", 8000))
    debug_mode = os.environ.get("FLASK_DEBUG", "True").lower() == "true"
    
    logger.info(f"🌐 Servidor iniciando na porta {port} (debug={debug_mode})")
    
    app.run(
        debug=debug_mode,
        host="0.0.0.0",
        port=port,
        threaded=True  # Permite múltiplas requisições simultâneas
    )

if __name__ == "__main__":
    main()
