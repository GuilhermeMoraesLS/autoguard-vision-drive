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
import concurrent.futures
from typing import List, Optional, Tuple, Dict, Any
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import statistics
import threading
import sys
import urllib.request

import cv2
import numpy as np
import requests
from PIL import Image
from cachetools import TTLCache
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from numpy.linalg import norm

# =====================================================
# CONFIGURAÇÃO E CONSTANTES OTIMIZADAS
# =====================================================

# Força UTF-8 no console (evita UnicodeEncodeError com emojis no Windows)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Configuração de logging segura para UTF-8
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # usa stdout/stderr reconfigurados acima
        logging.FileHandler("facial_recognition.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

class Config:
    # --- Qualidade / detecção ---
    PRELOAD_CASCADES = True
    MIN_FACE_AREA = 4000
    MIN_QUALITY_SCORE = 0.60
    REQUIRE_EYE_DETECTION = False

    # --- Similaridade / decisão (SFace + cosseno) ---
    MIN_DECISION_MARGIN = 0.06
    MIN_STRICT_THRESHOLD = 0.60   # ↓ aceitação “normal”
    MIN_LOOSE_THRESHOLD  = 0.52   # ↓ faixa de quase

    ENABLE_FALLBACK_RECHECK = True
    FALLBACK_THRESHOLD = 0.58     # recheck precisa atingir isso
    FALLBACK_MARGIN = 0.06

    # --- Embeddings ---
    FACE_SIZE = 160
    EMBEDDING_DIMENSIONS = 128     # SFace retorna 128D
    EMBEDDING_NORMALIZE = True
    EMBEDDING_TTA_FLIP = False     # não necessário com alinhamento SFace
    EMBEDDING_FP = np.float32
    ENABLE_FAST_MODE = True

    # --- Modelos OpenCV ---
    USE_SFACE = True
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
    YUNET_MODEL = "face_detection_yunet_2023mar.onnx"
    SFACE_MODEL = "face_recognition_sface_2021dec.onnx"
    YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

    # --- Paralelismo / cache ---
    MAX_PARALLEL_FACES = 4
    FEATURE_EXTRACTION_THREADS = 4
    REQUEST_TIMEOUT = 5
    CACHE_MAX_SIZE = 1024
    CACHE_TTL = 600
    # Haar (fallback)
    HAAR_PRIMARY = 'haarcascade_frontalface_alt2.xml'
    HAAR_FALLBACK = 'haarcascade_frontalface_default.xml'
    FACE_DET_SCALES = [1.08, 1.10, 1.20]
    FACE_DET_MIN_NEIGHBORS = [5, 4, 3]
    FACE_MIN_SIZE = (40, 40)
    FACE_MAX_SIZE = None

# Enums otimizados
class ValidationReason(Enum):
    VALID_FACE = "valid_face"
    FACE_TOO_SMALL = "face_too_small"
    INVALID_PROPORTIONS = "invalid_proportions"
    NO_EYES_DETECTED = "no_eyes_detected"
    TOO_UNIFORM = "too_uniform"
    LOW_GRADIENTS = "low_gradients"
    VALIDATION_ERROR = "validation_error"
    POOR_QUALITY = "poor_quality"
    NO_FACE_DETECTED = "no_face_detected"
    PROCESSING_ERROR = "processing_error"
    TOO_SMALL = "too_small"

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
# INICIALIZAÇÃO OTIMIZADA
# =====================================================

app = Flask(__name__)

def setup_cors():
    allowed_origins = [
        origin.strip() 
        for origin in os.environ.get("BACKEND_ALLOWED_ORIGINS", "").split(",") 
        if origin.strip()
    ]
    if not allowed_origins:
        allowed_origins = ["*"]
    
    CORS(app, resources={
        r"/*": {
            "origins": allowed_origins,
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    }, supports_credentials=True)

setup_cors()

# Caches otimizados
class OptimizedCache:
    def __init__(self):
        self.driver_embeddings = TTLCache(maxsize=Config.CACHE_MAX_SIZE, ttl=Config.CACHE_TTL)
        self.similarity = TTLCache(maxsize=4096, ttl=300)
        self.face_detections = TTLCache(maxsize=256, ttl=300)  # Cache de detecções (5 min)
        self.similarity_cache = TTLCache(maxsize=512, ttl=600)  # Cache de similaridades (10 min)
        self.lock = threading.RLock()
    
    def get_driver_embedding(self, driver_id: str) -> Optional[List[np.ndarray]]:
        with self.lock:
            return self.driver_embeddings.get(driver_id)
    
    def set_driver_embedding(self, driver_id: str, embeddings: List[np.ndarray]):
        with self.lock:
            self.driver_embeddings[driver_id] = embeddings
    
    def get_similarity(self, key: str) -> Optional[float]:
        with self.lock:
            return self.similarity_cache.get(key)
    
    def set_similarity(self, key: str, value: float):
        with self.lock:
            self.similarity_cache[key] = value

optimized_cache = OptimizedCache()
similarity_history: List[float] = []

# Classificadores globais pré-carregados
_face_cascade: Optional[cv2.CascadeClassifier] = None
_eye_cascade: Optional[cv2.CascadeClassifier] = None
_face_cascade_fallback: Optional[cv2.CascadeClassifier] = None
_cascade_lock = threading.Lock()

def preload_cascades():
    """Pré-carrega classificadores para melhor performance"""
    global _face_cascade, _eye_cascade, _face_cascade_fallback
    
    if not Config.PRELOAD_CASCADES:
        return
        
    with _cascade_lock:
        if _face_cascade is None:
            try:
                cascade_path = cv2.data.haarcascades + Config.HAAR_PRIMARY
                _face_cascade = cv2.CascadeClassifier(cascade_path)
                logger.info("✅ Face cascade (primário) pré-carregado")
            except Exception as e:
                logger.error(f"❌ Erro ao pré-carregar face cascade primário: {e}")
        # fallback
        if _face_cascade_fallback is None:
            try:
                cascade_path = cv2.data.haarcascades + Config.HAAR_FALLBACK
                _face_cascade_fallback = cv2.CascadeClassifier(cascade_path)
                logger.info("✅ Face cascade (fallback) pré-carregado")
            except Exception as e:
                logger.error(f"❌ Erro ao pré-carregar face cascade fallback: {e}")
        
        if _eye_cascade is None:
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
                _eye_cascade = cv2.CascadeClassifier(cascade_path)
                logger.info("✅ Eye cascade pré-carregado")
            except Exception as e:
                logger.error(f"❌ Erro ao pré-carregar eye cascade: {e}")

# Pré-carrega na inicialização
preload_cascades()

# =====================================================
# UTILITÁRIOS OTIMIZADOS
# =====================================================

def log_request_info():
    g.start_time = time.perf_counter()
    logger.info(f"📥 {request.method} {request.path}")

def log_response_info(response_data: dict):
    if hasattr(g, 'start_time'):
        duration = time.perf_counter() - g.start_time
        logger.info(f"📤 Resposta em {duration:.3f}s - Status: {response_data.get('status', 'unknown')}")

def create_error_response(message: str, status_code: int = 500, details: Any = None) -> tuple:
    error_response = {
        "status": "error",
        "message": message,
        "authorized": False
    }
    if details:
        error_response["details"] = details
    
    log_response_info(error_response)
    return jsonify(error_response), status_code

def ensure_json_safe(obj: Any) -> Any:
    """Converte recursivamente tipos NumPy/OpenCV para tipos nativos JSON-serializáveis."""
    import numpy as _np
    if isinstance(obj, dict):
        return {k: ensure_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ensure_json_safe(v) for v in obj]
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, _np.generic):  # np.int32, np.float32, etc.
        return obj.item()
    return obj

def create_success_response(data: dict, message: str = "Operação realizada com sucesso") -> tuple:
    success_response = {
        "status": "success",
        "message": message,
        **data
    }
    # Torna a árvore inteira serializável
    success_response = ensure_json_safe(success_response)
    log_response_info(success_response)
    return jsonify(success_response), 200

# =====================================================
# CLASSIFICADORES OTIMIZADOS
# =====================================================

def get_face_cascade() -> Optional[cv2.CascadeClassifier]:
    global _face_cascade
    if _face_cascade is None:
        preload_cascades()
    return _face_cascade

def get_face_cascade_fallback() -> Optional[cv2.CascadeClassifier]:
    global _face_cascade_fallback
    if _face_cascade_fallback is None:
        preload_cascades()
    return _face_cascade_fallback

def get_eye_cascade() -> Optional[cv2.CascadeClassifier]:
    global _eye_cascade
    if _eye_cascade is None:
        preload_cascades()
    return _eye_cascade

# =====================================================
# PROCESSAMENTO DE IMAGENS OTIMIZADO
# =====================================================

def validate_image_input(image_data: str) -> np.ndarray:
    """Valida e converte imagem base64 para array RGB (otimizado)"""
    try:
        if not image_data:
            raise ValueError("Dados de imagem não fornecidos")
        
        if "," in image_data:
            _, enc = image_data.split(",", 1)
        else:
            enc = image_data
        
        raw = base64.b64decode(enc)
        if len(raw) == 0:
            raise ValueError("Dados base64 vazios")
        
        # Otimização: usar cv2 diretamente é mais rápido que PIL
        nparr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Imagem inválida")
        
        # Converte BGR para RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if img.shape[0] < 100 or img.shape[1] < 100:
            raise ValueError(f"Imagem muito pequena: {img.shape}. Mínimo: 100x100")
        
        return img
        
    except Exception as e:
        logger.error(f"❌ Erro ao processar imagem: {e}")
        raise ValueError(f"Imagem base64 inválida: {str(e)}")

def validate_face_authenticity_fast(face_rgb: np.ndarray) -> ValidationResult:
    """Validação de face otimizada para velocidade"""
    try:
        h, w = face_rgb.shape[:2]
        area = h * w
        if area < Config.MIN_FACE_AREA:
            return ValidationResult(
                is_valid=False,
                reason=ValidationReason.FACE_TOO_SMALL,
                details={"area": int(area)},
                confidence=0.2
            )

        # Proporção
        aspect_ratio = w / max(h, 1)

        # Detecção de olhos (opcional): agora NÃO reprova, apenas impacta a confiança
        eyes_detected = 0
        if Config.REQUIRE_EYE_DETECTION:
            eye_cascade = get_eye_cascade()
            if eye_cascade is not None:
                face_gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
                eyes = eye_cascade.detectMultiScale(
                    face_gray, scaleFactor=1.2, minNeighbors=2, minSize=(15, 15)
                )
                eyes_detected = len(eyes)

        # Uniformidade/contraste
        face_gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
        std_dev = float(np.std(face_gray))

        if std_dev < 6:  # ligeiramente mais permissivo
            return ValidationResult(
                is_valid=False,
                reason=ValidationReason.TOO_UNIFORM,
                details=f"Std: {std_dev:.2f}",
                confidence=0.35
            )

        # Métrica de nitidez simples (gradiente)
        gy, gx = np.gradient(face_gray.astype(np.float32))
        sharpness = float(np.mean(gx**2 + gy**2))

        # Confiança composta
        confidence = 0.0
        confidence += min(1.0, (area / max(Config.MIN_FACE_AREA, 1)) * 0.35)
        confidence += max(0.0, (1.0 - abs(aspect_ratio - 1.0))) * 0.20
        confidence += min(1.0, std_dev / 30.0) * 0.25
        confidence += min(1.0, sharpness / 2000.0) * 0.20

        # Penalização leve se não detectar olhos (sem reprovar)
        if eyes_detected == 0:
            confidence *= 0.9

        return ValidationResult(
            is_valid=True,
            reason=ValidationReason.VALID_FACE,
            details={"area": int(area), "aspect_ratio": float(aspect_ratio), "std_dev": std_dev, "sharpness": sharpness, "eyes": eyes_detected},
            confidence=float(confidence)
        )
    except Exception as e:
        logger.error(f"❌ Validação de face falhou: {e}")
        return ValidationResult(
            is_valid=True,
            reason=ValidationReason.VALIDATION_ERROR,
            details=str(e),
            confidence=0.6
        )

def _align_with_yunet(rgb_img: np.ndarray) -> Optional[np.ndarray]:
    """
    Tenta alinhar o rosto com YuNet + SFace.alignCrop.
    Retorna imagem RGB alinhada ou None se falhar.
    """
    try:
        if not (Config.USE_SFACE and get_yunet_detector() is not None and get_sface_recognizer() is not None):
            return None
        bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        get_yunet_detector((bgr.shape[1], bgr.shape[0]))
        num, faces = _yunet_detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        face = faces[0]
        aligned = _sface_recognizer.alignCrop(bgr, face)
        if aligned is None or aligned.size == 0:
            return None
        return cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.debug(f"align_with_yunet falhou: {e}")
        return None

def _align_roi_with_yunet(rgb_img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
    """
    Tenta alinhar um ROI usando YuNet; se falhar retorna None.
    """
    try:
        if not (Config.USE_SFACE and get_yunet_detector() is not None and get_sface_recognizer() is not None):
            return None
        roi = rgb_img[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
        get_yunet_detector((bgr.shape[1], bgr.shape[0]))
        num, faces = _yunet_detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        aligned = _sface_recognizer.alignCrop(bgr, faces[0])
        if aligned is None or aligned.size == 0:
            return None
        return cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.debug(f"align_roi_with_yunet falhou: {e}")
        return None

def detect_and_validate_faces_parallel(rgb_image: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], np.ndarray, ValidationResult]]:
    """Detecção de faces otimizada com processamento paralelo"""
    try:
        # 1) Caminho preferencial: YuNet (+ landmarks) -> SFace.alignCrop
        if Config.USE_SFACE and get_yunet_detector() is not None and get_sface_recognizer() is not None:
            bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            get_yunet_detector((bgr.shape[1], bgr.shape[0]))
            try:
                num, faces = _yunet_detector.detect(bgr)
            except Exception as e:
                logger.warning(f"⚠️ YuNet detect falhou, caindo para Haar: {e}")
                num, faces = 0, None

            validated = []
            if faces is not None and len(faces) > 0:
                for face in faces:
                    x, y, w, h = face[:4].astype(int)
                    try:
                        aligned_bgr = _sface_recognizer.alignCrop(bgr, face)
                        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
                    except Exception:
                        # fallback: crop com margem
                        m = int(w * 0.15)
                        x1 = max(0, x - m); y1 = max(0, y - m)
                        x2 = min(bgr.shape[1], x + w + m); y2 = min(bgr.shape[0], y + h + m)
                        aligned_rgb = rgb_image[y1:y2, x1:x2]

                    if aligned_rgb.size == 0:
                        continue

                    validation = validate_face_authenticity_fast(aligned_rgb)
                    if validation.is_valid:
                        validated.append(((int(x), int(y), int(w), int(h)), aligned_rgb, validation))

                if len(validated) > 0:
                    logger.info(f"✅ {len(validated)} faces válidas (YuNet/SFace)")
                    return validated
            # Se nada válido com YuNet, continua no fallback Haar

        # 2) Fallback Haar (mantém sua lógica atual), mas tenta alinhar ROI com YuNet antes
        face_cascade = get_face_cascade()
        face_cascade_fb = get_face_cascade_fallback()
        if face_cascade is None and face_cascade_fb is None:
            return []
        
        # Redimensiona imagem se muito grande (otimização)
        original_shape = rgb_image.shape
        max_size = 1280  # um pouco maior para manter rostos grandes
        if max(original_shape[:2]) > max_size:
            scale = max_size / max(original_shape[:2])
            new_size = (int(original_shape[1] * scale), int(original_shape[0] * scale))
            rgb_resized = cv2.resize(rgb_image, new_size)
            scale_back = 1 / scale
        else:
            rgb_resized = rgb_image
            scale_back = 1.0
        
        gray = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2GRAY)

        faces = []
        cascades = [c for c in [face_cascade, face_cascade_fb] if c is not None]
        found = False
        for cas_idx, cascade in enumerate(cascades):
            for sf in Config.FACE_DET_SCALES:
                for mn in Config.FACE_DET_MIN_NEIGHBORS:
                    params = dict(scaleFactor=sf, minNeighbors=mn, flags=cv2.CASCADE_SCALE_IMAGE)
                    if Config.FACE_MIN_SIZE:
                        params["minSize"] = Config.FACE_MIN_SIZE
                    if Config.FACE_MAX_SIZE:
                        params["maxSize"] = Config.FACE_MAX_SIZE

                    faces = cascade.detectMultiScale(gray, **params)
                    logger.info(f"🔎 Haar[{cas_idx}] sf={sf} mn={mn} -> faces={len(faces)}")
                    if len(faces) > 0:
                        found = True
                        break
                if found: break
            if found: break

        if len(faces) == 0:
            logger.warning("❌ Nenhuma face detectada (todas as tentativas).")
            return []

        # Escala coordenadas de volta
        if scale_back != 1.0:
            faces = [(int(x * scale_back), int(y * scale_back), int(w * scale_back), int(h * scale_back)) for x, y, w, h in faces]
        else:
            faces = [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]

        validated_faces = []
        faces_to_process = faces[:Config.MAX_PARALLEL_FACES]

        def process_single_face(face_coords):
            x, y, w, h = face_coords
            margin = int(w * 0.15)
            x1 = max(0, x - margin); y1 = max(0, y - margin)
            x2 = min(rgb_image.shape[1], x + w + margin); y2 = min(rgb_image.shape[0], y + h + margin)

            # Tenta alinhar o ROI com YuNet; se falhar usa o crop simples
            aligned_rgb = _align_roi_with_yunet(rgb_image, x1, y1, x2, y2)
            face_rgb = aligned_rgb if aligned_rgb is not None else rgb_image[y1:y2, x1:x2]

            if face_rgb.size > 0:
                validation = validate_face_authenticity_fast(face_rgb)
                if validation.is_valid:
                    return ((x, y, w, h), face_rgb, validation)
                else:
                    logger.info(f"⚠️ Face rejeitada na validação: {validation.reason} conf={validation.confidence:.2f}")
            return None

        if Config.FEATURE_EXTRACTION_THREADS > 1 and len(faces_to_process) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=Config.FEATURE_EXTRACTION_THREADS) as executor:
                results = list(executor.map(process_single_face, faces_to_process))
                validated_faces = [r for r in results if r is not None]
        else:
            for fc in faces_to_process:
                r = process_single_face(fc)
                if r: validated_faces.append(r)

        logger.info(f"✅ {len(validated_faces)} faces válidas de {len(faces)} detectadas (fallback Haar)")
        return validated_faces
    except Exception as e:
        logger.error(f"❌ Erro na detecção de faces: {e}\n{traceback.format_exc()}")
        return []

def assess_face_quality_fast(face_rgb: np.ndarray) -> QualityMetrics:
    """Avaliação de qualidade otimizada"""
    try:
        face_gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
        
        # Brilho
        brightness = np.mean(face_gray) / 255.0
        brightness_score = max(0, 1.0 - abs(brightness - 0.5) * 2)
        
        # Contraste
        contrast = np.std(face_gray) / 128.0
        contrast_score = min(contrast, 1.0)
        
        # Nitidez simplificada
        if Config.ENABLE_FAST_MODE:
            # Usa Sobel em vez de Laplaciano (mais rápido)
            grad_x = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)
            sharpness_var = np.mean(grad_x**2 + grad_y**2)
            sharpness_score = min(sharpness_var / 5000.0, 1.0)
        else:
            laplacian_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 1000.0, 1.0)
        
        # Resolução
        resolution_score = min(min(face_rgb.shape[:2]) / 80.0, 1.0)  # Menor exigência
        
        # Uniformidade
        uniformity = 1.0 - min(np.std(face_gray) / 64.0, 1.0)
        uniformity_score = max(0, 1.0 - uniformity)
        
        # Score geral com pesos otimizados
        weights = [0.15, 0.3, 0.3, 0.15, 0.1]  # Prioriza contraste e nitidez
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
# EXTRAÇÃO DE EMBEDDINGS OTIMIZADA
# =====================================================

class OptimizedEmbeddingExtractor:
    """Extrator de embeddings otimizado para velocidade"""
    
    @staticmethod
    def extract_fast_features(face_gray: np.ndarray) -> np.ndarray:
        """Extrai features rapidamente com menos detalhes"""
        features = []
        
        # 1. Histograma reduzido
        hist = cv2.calcHist([face_gray], [0], None, [32], [0, 256])  # Reduzido de 64 para 32
        features.extend(hist.flatten())
        
        # 2. Grid simplificado 8x8 (em vez de 16x16)
        grid_size = 8
        cell_h = Config.FACE_SIZE // grid_size
        cell_w = Config.FACE_SIZE // grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                y_start = i * cell_h
                y_end = (i + 1) * cell_h
                x_start = j * cell_w
                x_end = (j + 1) * cell_w
                
                cell = face_gray[y_start:y_end, x_start:x_end]
                
                # Apenas estatísticas básicas
                features.extend([
                    np.mean(cell),
                    np.std(cell),
                    np.min(cell),
                    np.max(cell)
                ])
        
        # 3. Gradientes simplificados
        grad_x = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)
        
        features.extend([
            np.mean(np.abs(grad_x)),
            np.mean(np.abs(grad_y)),
            np.mean(np.sqrt(grad_x**2 + grad_y**2)),
            np.std(grad_x),
            np.std(grad_y)
        ])
        
        # 4. LBP simplificado
        if not Config.ENABLE_FAST_MODE:
            lbp = OptimizedEmbeddingExtractor._calculate_lbp_fast(face_gray)
            lbp_hist = np.histogram(lbp, bins=16, range=(0, 255))[0]  # Reduzido de 32 para 16
            features.extend(lbp_hist)
        else:
            # Pula LBP no modo rápido
            features.extend([0.0] * 16)
        
        # 5. Características de simetria simplificadas
        h, w = face_gray.shape
        left_half = face_gray[:, :w//2]
        right_half = np.fliplr(face_gray[:, w//2:])
        
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        symmetry_diff = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))
        features.append(symmetry_diff)
        
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def _calculate_lbp_fast(image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """LBP otimizado usando operações vetorizadas"""
        lbp = np.zeros_like(image)
        
        # Versão mais rápida usando slicing
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            dy = int(radius * np.sin(angle))
            dx = int(radius * np.cos(angle))
            
            # Calcula deslocamentos seguros
            y_min = max(0, -dy)
            y_max = min(image.shape[0], image.shape[0] - dy)
            x_min = max(0, -dx)
            x_max = min(image.shape[1], image.shape[1] - dx)
            
            center_region = image[y_min:y_max, x_min:x_max]
            neighbor_region = image[y_min+dy:y_max+dy, x_min+dx:x_max+dx]
            
            # Operação vetorizada
            lbp[y_min:y_max, x_min:x_max] += (neighbor_region >= center_region) * (2 ** i)
        
        return lbp
    
    @staticmethod
    def normalize_embedding_fast(embedding: np.ndarray) -> Optional[np.ndarray]:
        """Normalização otimizada"""
        try:
            # Padding/truncate otimizado
            target_size = Config.EMBEDDING_DIMENSIONS
            if len(embedding) < target_size:
                # Repetição simples sem ruído adicional
                repeats = target_size // len(embedding) + 1
                embedding = np.tile(embedding, repeats)[:target_size]
            else:
                embedding = embedding[:target_size]
            
            # Normalização L2 otimizada
            embedding = embedding.astype(np.float32)
            norm_val = np.linalg.norm(embedding)
            
            if norm_val > 1e-12:
                embedding = embedding / norm_val
            else:
                logger.error("❌ Embedding com norma zero!")
                return None
            
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Erro na normalização: {e}")
            return None

def extract_embedding_optimized(face_rgb: np.ndarray) -> np.ndarray | None:
    """
    Com SFace: usa alignCrop previamente e extrai vetor 128D; normaliza L2.
    Fallback: pipeline antigo.
    """
    try:
        # SFace espera BGR; aqui face_rgb já vem alinhado da detecção YuNet
        if Config.USE_SFACE and get_sface_recognizer() is not None:
            bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
            feat = _sface_recognizer.feature(bgr)  # (1,128) float32
            if feat is None:
                return None
            emb = np.asarray(feat).reshape(-1).astype(Config.EMBEDDING_FP)
            if Config.EMBEDDING_NORMALIZE:
                emb = l2_normalize(emb)
            return emb

        # ===== Fallback para o extrator antigo =====
        proc = preprocess_face_for_embedding(face_rgb)
        emb1 = _infer_embedding_np(proc)
        if emb1 is None:
            return None
        emb = emb1.astype(Config.EMBEDDING_FP)
        if Config.EMBEDDING_NORMALIZE:
            emb = l2_normalize(emb)
        return emb
    except Exception as e:
        logger.error(f"❌ Erro em extract_embedding_optimized: {e}")
        return None

# =====================================================
# ALGORITMOS DE SIMILARIDADE OTIMIZADOS
# =====================================================

def calculate_cosine_similarity_cached(v1: np.ndarray, v2: np.ndarray, cache_key: str = None) -> float:
    """Similaridade cosseno com cache"""
    try:
        if cache_key:
            cached = optimized_cache.get_similarity(cache_key)
            if cached is not None:
                return cached
        
        # Cálculo otimizado
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norm_product < 1e-12:
            similarity = 0.0
        else:
            similarity = float(dot_product / norm_product)
        
        if cache_key:
            optimized_cache.set_similarity(cache_key, similarity)
        
        return similarity
        
    except Exception as e:
        logger.error(f"❌ Erro no cálculo de similaridade: {e}")
        return 0.0

def calculate_dynamic_thresholds_fast() -> tuple[float, float]:
    """
    Para SFace + cosseno (L2), mantém pisos adequados.
    """
    try:
        if len(similarity_history) < 10:
            return max(0.80, Config.MIN_STRICT_THRESHOLD), max(0.60, Config.MIN_LOOSE_THRESHOLD)
        recent = similarity_history[-100:]
        mean = float(np.mean(recent))
        std = float(np.std(recent))
        strict = max(Config.MIN_STRICT_THRESHOLD, min(0.95, mean + 0.5 * std))
        loose  = max(Config.MIN_LOOSE_THRESHOLD,  min(0.85, mean - 0.2 * std))
        return strict, loose
    except Exception:
        return Config.MIN_STRICT_THRESHOLD, Config.MIN_LOOSE_THRESHOLD

# =====================================================
# PROCESSAMENTO DE MOTORISTAS OTIMIZADO
# =====================================================

def _cache_key_driver(did: str, url: str) -> str:
    return f"drv:{did}:{url}"

def get_cached_driver_embedding(did: str, url: str) -> Optional[np.ndarray]:
    try:
        return optimized_cache.driver_embeddings.get(_cache_key_driver(did, url))
    except Exception:
        return None

def set_cached_driver_embedding(did: str, url: str, emb: np.ndarray) -> None:
    try:
        optimized_cache.driver_embeddings[_cache_key_driver(did, url)] = emb
    except Exception:
        pass

def process_driver_embeddings_parallel(authorized_drivers) -> Tuple[List[np.ndarray], List[Tuple[str, str]]]:
    """Constrói (known_embeddings, id_name_pairs) com template robusto por motorista e cache."""
    known_embeddings: List[np.ndarray] = []
    id_name_pairs: List[Tuple[str, str]] = []

    def build_one(drv):
        did = str(drv["id"])
        url = str(drv["photo_url"])
        cached = get_cached_driver_embedding(did, url)
        if cached is not None:
            return (cached, (did, str(drv["name"])))

        emb = compute_driver_template_embedding(url)
        if emb is None:
            logger.warning(f"⚠️ embedding vazio para {drv.get('name')}")
            return None
        emb = l2_normalize(emb)
        set_cached_driver_embedding(did, url, emb)
        return (emb, (did, str(drv["name"])))

    results: List[Optional[Tuple[np.ndarray, Tuple[str,str]]]] = []
    if Config.FEATURE_EXTRACTION_THREADS > 1 and len(authorized_drivers) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=Config.FEATURE_EXTRACTION_THREADS) as ex:
            results = list(ex.map(build_one, authorized_drivers))
    else:
        for d in authorized_drivers:
            results.append(build_one(d))

    for r in results:
        if r is None:
            continue
        emb, pair = r
        known_embeddings.append(l2_normalize(emb))
        id_name_pairs.append(pair)

    logger.info(f"✅ embeddings conhecidos (cache aware): {len(known_embeddings)}")
    return known_embeddings, id_name_pairs

def build_known_embeddings(authorized_drivers_payload) -> tuple[list[np.ndarray], list[tuple[str, str]]]:
    """
    Retorna (known_embeddings, id_name_pairs) com embeddings já L2-normalizados.
    """
    known_embeddings: list[np.ndarray] = []
    id_name_pairs: list[tuple[str, str]] = []
    for drv in authorized_drivers_payload:
        # ...existing code para baixar/cortar face e extrair embedding -> emb...
        emb = extract_embedding_optimized(face_rgb)
        if emb is None:
            continue
        emb = l2_normalize(emb)
        known_embeddings.append(emb)
        id_name_pairs.append((drv["id"], drv["name"]))
    return known_embeddings, id_name_pairs

# =====================================================
# ROTAS DA API OTIMIZADAS
# =====================================================

@app.before_request
def before_request():
    log_request_info()

@app.route("/health", methods=["GET"])
def health_check():
    """Health check otimizado"""
    try:
        health_data = {
            "face_detection": "✅ YuNet" if (Config.USE_SFACE and get_yunet_detector() is not None) else ("✅ Haar" if get_face_cascade() else "❌"),
            "eye_detection": "✅" if get_eye_cascade() else "❌",
            "embedding_method": ("SFace (128D)" if Config.USE_SFACE and get_sface_recognizer() is not None else f"Features otimizadas ({Config.EMBEDDING_DIMENSIONS}D)"),
            "performance_mode": "🚀 Modo rápido" if Config.ENABLE_FAST_MODE else "🔍 Modo preciso",
            "cache_stats": {
                "driver_embeddings": len(optimized_cache.driver_embeddings),
                "similarity_cache": len(optimized_cache.similarity_cache),
                "face_detections": len(optimized_cache.face_detections)
            },
            "config": {
                "parallel_faces": Config.MAX_PARALLEL_FACES,
                "threads": Config.FEATURE_EXTRACTION_THREADS,
                "face_size": Config.FACE_SIZE,
            }
        }
        return create_success_response(health_data, "API otimizada online!")
    except Exception as e:
        logger.error(f"❌ Health check erro: {e}")
        return create_error_response("Falha no health", 500)

# Adicione esta função antes do endpoint verify_driver_optimized:

def calculate_face_distance(face1_coords, face2_coords):
    """Calcula a distância entre duas faces para evitar duplicações"""
    x1, y1, w1, h1 = face1_coords
    x2, y2, w2, h2 = face2_coords
    
    # Centro das faces
    center1 = (x1 + w1//2, y1 + h1//2)
    center2 = (x2 + w2//2, y2 + h2//2)
    
    # Distância euclidiana
    distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    return distance

def remove_duplicate_faces(faces, min_distance=50):
    """Remove faces duplicadas ou muito próximas"""
    if len(faces) <= 1:
        return faces
    
    filtered_faces = []
    
    for i, face1 in enumerate(faces):
        coords1, crop1, validation1 = face1
        is_duplicate = False
        
        for j, face2 in enumerate(filtered_faces):
            coords2, crop2, validation2 = face2
            distance = calculate_face_distance(coords1, coords2)
            
            if distance < min_distance:
                # Se muito próximas, mantém a com melhor confiança
                if validation1.confidence > validation2.confidence:
                    filtered_faces[j] = face1
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered_faces.append(face1)
    
    return filtered_faces

@app.route("/verify_driver", methods=["POST"])
def verify_driver_optimized():
    """Endpoint principal OTIMIZADO de verificação"""
    try:
        # Validação rápida
        payload = request.get_json(silent=True)
        if not payload:
            return create_error_response("Payload JSON inválido", 400)

        image_data = payload.get("image")
        authorized_drivers = payload.get("authorized_drivers", [])  
        car_id = payload.get("car_id", "N/A")

        if not image_data or not isinstance(authorized_drivers, list):
            return create_error_response("Dados inválidos", 400)

        logger.info(f"🚗 Verificação OTIMIZADA - Carro {car_id}, {len(authorized_drivers)} motoristas")

        # Processamento de imagem otimizado
        try:
            rgb_image = validate_image_input(image_data)
            logger.info(f"🖼️ Imagem processada: {rgb_image.shape}")
        except ValueError as e:
            return create_error_response(str(e), 400)

        # Processamento paralelo de motoristas
        known_embeddings, id_name_pairs = process_driver_embeddings_parallel(authorized_drivers)
        if not known_embeddings:
            # thresholds default para manter contrato de resposta
            th_strict, th_loose = calculate_dynamic_thresholds_fast()
            return create_success_response({
                "detections": [],                 # <- sempre presente
                "car_id": car_id,
                "authorized_count": 0,
                "unknown_count": 0,
                "thresholds": {"strict": th_strict, "loose": th_loose},
                "image_dimensions": {"width": rgb_image.shape[1], "height": rgb_image.shape[0]},
            }, "Nenhum motorista autorizado válido processado.")

        # Detecção paralela de faces
        all_detected_faces = detect_and_validate_faces_parallel(rgb_image)
        all_detected_faces = remove_duplicate_faces(all_detected_faces, min_distance=80)

        # known_embeddings e id_name_pairs já existem aqui
        # GARANTE ALINHAMENTO EXPLÍCITO (id, name, embedding) por índice
        known = []
        for i in range(len(known_embeddings)):
            did, dname = id_name_pairs[i]
            emb = known_embeddings[i]
            known.append((did, dname, emb))

        detection_results = []
        used_driver_ids = set()

        th_strict, th_loose = calculate_dynamic_thresholds_fast()
        # ...existing code...

        for face_index, ((x, y, w, h), face_crop, validation) in enumerate(all_detected_faces):
            query_emb = extract_embedding_optimized(face_crop)
            if query_emb is None:
                detection_results.append({
                    "authorized": False, "driver_id": None,
                    "driver_name": f"Desconhecido #{int(face_index)+1}",
                    "confidence": float(0.0),
                    "x": int(x), "y": int(y), "width": int(w), "height": int(h)
                })
                continue

            query_emb = l2_normalize(query_emb)

            sims: list[float] = []
            candidates: list[tuple[str, str]] = []
            for i, known_emb in enumerate(known_embeddings):
                did, dname = id_name_pairs[i]
                candidates.append((did, dname))
                known_norm = l2_normalize(known_emb)
                sim = calculate_cosine_similarity_cached(query_emb, known_norm, cache_key=f"{did}_{face_index}")
                sims.append(float(sim))

            if not sims:
                detection_results.append({
                    "authorized": False, "driver_id": None,
                    "driver_name": f"Desconhecido #{int(face_index)+1}",
                    "confidence": float(0.0),
                    "x": int(x), "y": int(y), "width": int(w), "height": int(h)
                })
                continue

            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            second_best = float(sorted(sims)[-2] if len(sims) >= 2 else 0.0)
            margin = float(best_sim - second_best)
            best_driver_id, best_driver_name = candidates[best_idx]

            # Caminho estrito
            strict_ok = (
                best_sim >= float(th_strict) and
                margin >= float(Config.MIN_DECISION_MARGIN) and
                validation.is_valid and
                float(validation.confidence) >= float(Config.MIN_QUALITY_SCORE) and
                best_driver_id not in used_driver_ids
            )

            # Caminho fallback “no-quase”: rechecagem robusta + margem maior
            fallback_ok = False
            if (Config.ENABLE_FALLBACK_RECHECK
                and not strict_ok
                and best_sim >= float(th_loose)
                and margin >= float(Config.FALLBACK_MARGIN)
                and validation.is_valid
                and float(validation.confidence) >= float(Config.MIN_QUALITY_SCORE)):
                re_sim = robust_recheck_similarity(face_crop, known_embeddings[best_idx])
                logger.info(f"🔁 Recheck sim={re_sim:.3f} (best={best_sim:.3f}, loose={th_loose:.3f})")
                fallback_ok = re_sim >= float(Config.FALLBACK_THRESHOLD)

            authorized = (strict_ok or fallback_ok)

            final_id = best_driver_id if authorized else None
            final_name = best_driver_name if authorized else f"Desconhecido #{int(face_index)+1}"
            if authorized:
                used_driver_ids.add(best_driver_id)

            # DEBUG INFO
            debug = {
                "best_sim": float(best_sim),
                "second_best": float(second_best),
                "margin": float(margin),
                "th_strict": float(th_strict),
                "th_loose": float(th_loose),
                "quality": float(validation.confidence),
                "fallback_used": bool(not strict_ok and authorized)
            }

            detection_results.append({
                "authorized": bool(authorized),
                "driver_id": final_id,
                "driver_name": final_name,
                "confidence": float(round(best_sim * 100.0, 1)),
                "x": int(x), "y": int(y), "width": int(w), "height": int(h),
                "debug": debug,
            })
        authorized_count = int(sum(1 for r in detection_results if r["authorized"]))
        unknown_count = int(len(detection_results) - authorized_count)

        return create_success_response({
            "detections": detection_results,
            "car_id": str(car_id),
            "authorized_count": authorized_count,
            "unknown_count": unknown_count,
            "thresholds": {"strict": float(th_strict), "loose": float(th_loose)},
            "image_dimensions": {"width": int(rgb_image.shape[1]), "height": int(rgb_image.shape[0])},
        }, f"Verificação: {authorized_count} autorizado(s), {unknown_count} desconhecido(s)")
    except Exception as e:
        logger.error(f"❌ Erro na verificação: {e}\n{traceback.format_exc()}")
        return create_error_response(f"Erro interno: {str(e)}")

@app.route("/analytics", methods=["GET"])
def get_analytics():
    """Endpoint para monitorar desempenho anti-falso positivo"""
    try:
        strict, loose = calculate_dynamic_thresholds_fast()
        return jsonify({
            "total_verifications": len(similarity_history),
            "average_similarity": statistics.mean(similarity_history) if similarity_history else 0,
            "current_thresholds": {"strict": strict, "loose": loose},
            "rejection_rate": (
                len([s for s in similarity_history if s < 0.7]) / len(similarity_history)
            ) if similarity_history else 0
        })
    except Exception as e:
        logger.error(f"❌ Erro no analytics: {e}\n{traceback.format_exc()}")
        return create_error_response("Erro ao coletar analytics")

@app.errorhandler(404)
def not_found(error):
    return create_error_response("Endpoint não encontrado", 404)

# =====================================================
# INICIALIZAÇÃO OTIMIZADA
# =====================================================

# Modelos SFace / YuNet
_sface_recognizer: Optional[any] = None
_yunet_detector: Optional[any] = None

def _ensure_model_file(filename: str, url: str) -> str:
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    path = os.path.join(Config.MODEL_DIR, filename)
    if not os.path.exists(path):
        try:
            logger.info(f"⬇️ Baixando modelo {filename}...")
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
            logger.info(f"✅ Modelo salvo em {path}")
        except Exception as e:
            logger.error(f"❌ Falha ao baixar {filename}: {e}")
    return path

def get_sface_recognizer():
    global _sface_recognizer
    if not Config.USE_SFACE:
        return None
    if _sface_recognizer is None:
        try:
            model_path = _ensure_model_file(Config.SFACE_MODEL, Config.SFACE_URL)
            # config vazio para SFace
            _sface_recognizer = cv2.FaceRecognizerSF.create(model_path, "")
            logger.info("✅ SFace carregado")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar SFace: {e}")
            _sface_recognizer = None
    return _sface_recognizer

def get_yunet_detector(input_size: Tuple[int, int] | None = None):
    global _yunet_detector
    if not Config.USE_SFACE:
        return None
    if _yunet_detector is None:
        try:
            model_path = _ensure_model_file(Config.YUNET_MODEL, Config.YUNET_URL)
            _yunet_detector = cv2.FaceDetectorYN.create(model_path, "", (320, 320), 0.85, 0.3, 5000)
            logger.info("✅ YuNet carregado")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar YuNet: {e}")
            _yunet_detector = None
    if _yunet_detector is not None and input_size is not None:
        try:
            _yunet_detector.setInputSize(input_size)
        except Exception:
            pass
    return _yunet_detector

def initialize_optimized_system():
    """Inicialização otimizada do sistema"""
    try:
        preload_cascades()
        if Config.USE_SFACE:
            get_sface_recognizer()
            get_yunet_detector((640, 480))
        logger.info("🚀 AutoGuard Vision Backend - MODO OTIMIZADO")
        logger.info(f"⚡ Performance: Faces paralelas={Config.MAX_PARALLEL_FACES}, Threads={Config.FEATURE_EXTRACTION_THREADS}")
        logger.info(f"🧠 Embeddings: {Config.EMBEDDING_DIMENSIONS}D, Face size={Config.FACE_SIZE}x{Config.FACE_SIZE}")
        logger.info(f"🔧 Modo rápido: {'✅ Ativado' if Config.ENABLE_FAST_MODE else '❌ Desativado'}")
        logger.info("✅ Sistema de reconhecimento facial otimizado pronto!")
    except Exception as e:
        logger.error(f"❌ Erro na inicialização: {e}")

def main():
    """Função principal otimizada"""
    initialize_optimized_system()
    port = int(os.environ.get("PORT", 8000))
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    logger.info(f"🌐 Servidor OTIMIZADO na porta {port}")
    app.run(debug=debug_mode, host="0.0.0.0", port=port, threaded=True, use_reloader=False)

# =====================================================
# FUNÇÕES AUXILIARES (MANTENHA NO FINAL)
# =====================================================

def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    try:
        v = v.astype(Config.EMBEDDING_FP, copy=False)
        n = np.linalg.norm(v)
        if n < eps:
            return np.zeros_like(v, dtype=Config.EMBEDDING_FP)
        return (v / n).astype(Config.EMBEDDING_FP, copy=False)
    except Exception:
        return v

def preprocess_face_for_embedding(face_rgb: np.ndarray) -> np.ndarray:
    """
    - equalização (CLAHE no canal Y)
    - alinhamento leve por olhos (se disponíveis via cascade)
    - resize e normalização [0,1], pad mantendo aspecto
    """
    img = face_rgb.copy()
    try:
        # Alinhamento básico por olhos
        if Config.REQUIRE_EYE_DETECTION:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            eye_cascade = get_eye_cascade()
            if eye_cascade is not None:
                eyes = eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(15, 15))
                if len(eyes) >= 2:
                    eyes = sorted(eyes, key=lambda e: e[0])[:2]
                    (x1, y1, w1, h1), (x2, y2, w2, h2) = eyes[0], eyes[1]
                    p1 = np.array([x1 + w1 / 2.0, y1 + h1 / 2.0])
                    p2 = np.array([x2 + w2 / 2.0, y2 + h2 / 2.0])
                    dy, dx = (p2 - p1)[1], (p2 - p1)[0]
                    angle = np.degrees(np.arctan2(dy, dx))
                    M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1.0)
                    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

        # CLAHE no canal Y
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y = clahe.apply(y)
        img = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2RGB)

        # Pad para quadrado antes do resize (mantém aspecto)
        h, w = img.shape[:2]
        side = max(h, w)
        pad_t = (side - h) // 2
        pad_b = side - h - pad_t
        pad_l = (side - w) // 2
        pad_r = side - w - pad_l
        img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101)

        # Resize e normalização
        img = cv2.resize(img, (Config.FACE_SIZE, Config.FACE_SIZE), interpolation=cv2.INTER_AREA)
        img = img.astype(Config.EMBEDDING_FP) / 255.0
        # Normalização padrão (zero-mean unit-var por canal)
        mean = np.array([0.5, 0.5, 0.5], dtype=Config.EMBEDDING_FP)
        std = np.array([0.5, 0.5, 0.5], dtype=Config.EMBEDDING_FP)
        img = (img - mean) / (std + 1e-6)
        return img
    except Exception:
        return cv2.resize(face_rgb, (Config.FACE_SIZE, Config.FACE_SIZE)).astype(Config.EMBEDDING_FP) / 255.0

def _infer_embedding_np(img_norm: np.ndarray) -> np.ndarray | None:
    """
    Gera embedding 1D a partir da imagem normalizada (FACE_SIZE x FACE_SIZE x 3, float32).
    Usa o extrator rápido baseado em estatísticas/gradientes e normaliza para o tamanho alvo.
    """
    try:
        # img_norm foi normalizada com mean=0.5, std=0.5 -> volta para [0,255] para extrair cinza estável
        img01 = np.clip((img_norm * 0.5 + 0.5), 0.0, 1.0)
        img_u8 = (img01 * 255.0).astype(np.uint8)
        gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)

        feat = OptimizedEmbeddingExtractor.extract_fast_features(gray)
        emb = OptimizedEmbeddingExtractor.normalize_embedding_fast(feat)
        return emb
    except Exception as e:
        logger.error(f"❌ _infer_embedding_np failed: {e}")
        return None

def robust_recheck_similarity(face_rgb: np.ndarray, known_emb: np.ndarray) -> float:
    """
    Rechecagem robusta:
      - usa extract_embedding_optimized (com flip TTA)
      - aplica pequenas perturbações (blur leve, brilho)
      - retorna média das similaridades contra known_emb (L2-normalizados)
    """
    try:
        variants = [face_rgb]
        # blur leve
        variants.append(cv2.GaussianBlur(face_rgb, (3, 3), 0))
        # ajuste de brilho/contraste leve
        brighter = np.clip(face_rgb.astype(np.float32) * 1.06, 0, 255).astype(np.uint8)
        variants.append(brighter)

        sims = []
        known_emb = l2_normalize(known_emb)
        for v in variants:
            q = extract_embedding_optimized(v)
            if q is None:
                continue
            q = l2_normalize(q)
            sims.append(calculate_cosine_similarity_cached(q, known_emb, cache_key=None))
        if not sims:
            return 0.0
        return float(np.mean(sims))
    except Exception as e:
        logger.error(f"❌ robust_recheck_similarity failed: {e}")
        return 0.0

def _download_rgb_image(url: str) -> Optional[np.ndarray]:
    try:
        with urllib.request.urlopen(url, timeout=Config.REQUEST_TIMEOUT) as resp:
            data = np.frombuffer(resp.read(), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.error(f"❌ download image failed: {e}")
        return None

def _detect_biggest_face(rgb: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    try:
        cas = get_face_cascade() or get_face_cascade_fallback()
        if cas is None:
            return None
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = cas.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40,40)
        )
        if len(faces) == 0:
            return None
        faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
        x,y,w,h = map(int, faces[0])
        m = int(w*0.15)
        x1,y1 = max(0,x-m), max(0,y-m)
        x2,y2 = min(rgb.shape[1], x+w+m), min(rgb.shape[0], y+h+m)
        return (x1,y1,x2-x1,y2-y1)
    except Exception as e:
        logger.error(f"❌ detect biggest face failed: {e}")
        return None

def compute_driver_template_embedding(photo_url: str) -> Optional[np.ndarray]:
    """
    Template robusto: detecta com YuNet, alinha com SFace e faz média de variações.
    """
    rgb = _download_rgb_image(photo_url)
    if rgb is None:
        return None

    # tenta YuNet para obter alinhado
    emb_list = []
    if Config.USE_SFACE and get_yunet_detector() is not None and get_sface_recognizer() is not None:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        get_yunet_detector((bgr.shape[1], bgr.shape[0]))
        try:
            num, faces = _yunet_detector.detect(bgr)
        except Exception:
            num, faces = 0, None
        if faces is not None and len(faces) > 0:
            face = faces[0]
            try:
                aligned = _sface_recognizer.alignCrop(bgr, face)
                aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            except Exception:
                aligned_rgb = rgb
        else:
            aligned_rgb = rgb
    else:
        aligned_rgb = rgb

    variants = [aligned_rgb]
    try:
        variants.append(cv2.GaussianBlur(aligned_rgb, (3,3), 0))
        variants.append(np.clip(aligned_rgb.astype(np.float32)*1.05,0,255).astype(np.uint8))
        variants.append(np.clip(aligned_rgb.astype(np.float32)*0.95,0,255).astype(np.uint8))
    except Exception:
        pass

    for v in variants:
        e = extract_embedding_optimized(v)
        if e is not None:
            emb_list.append(l2_normalize(e))

    if not emb_list:
        return None
    tpl = np.mean(np.stack(emb_list, axis=0), axis=0).astype(Config.EMBEDDING_FP)
    return l2_normalize(tpl)

def compute_display_confidence(best_sim: float, second_best: float, th_loose: float, th_strict: float,
                               quality: float, used_fallback: bool, re_sim: float | None) -> float:
    """
    Converte similaridade (coseno) em % amigável:
    - < th_loose  -> 0–50%
    - th_loose–th_strict -> 50–80%
    - >= th_strict -> 80–100%
    Ajuste leve por margem e qualidade. Se fallback foi usado, considera re_sim.
    """
    base_sim = max(best_sim, re_sim or 0.0) if used_fallback else best_sim

    # mapeamento por faixas
    if base_sim <= th_loose:
        pct = 50.0 * max(0.0, base_sim / max(th_loose, 1e-6))  # 0..50
    elif base_sim < th_strict:
        pct = 50.0 + 30.0 * ((base_sim - th_loose) / max(th_strict - th_loose, 1e-6))  # 50..80
    else:
        pct = 80.0 + 20.0 * ((base_sim - th_strict) / max(1.0 - th_strict, 1e-6))  # 80..100

    # bônus por margem e qualidade
    margin = max(0.0, best_sim - second_best)
    margin_bonus = min(10.0, (margin / max(Config.MIN_DECISION_MARGIN, 1e-6)) * 4.0)  # até +10
    quality_bonus = min(5.0, max(0.0, (quality - Config.MIN_QUALITY_SCORE) / max(1.0 - Config.MIN_QUALITY_SCORE, 1e-6)) * 5.0)

    pct = min(100.0, max(0.0, pct + margin_bonus + quality_bonus))
    return round(pct, 1)

def build_similarity_matrix(face_embs: list[np.ndarray | None], known_embs: list[np.ndarray]) -> np.ndarray:
    """
    Retorna matriz S (n_faces x n_known) com cossenos (L2); linhas com None viram 0.
    """
    if not face_embs or not known_embs:
        return np.zeros((len(face_embs), len(known_embs)), dtype=np.float32)
    # normaliza todos
    K = np.stack([l2_normalize(e) for e in known_embs], axis=1)  # (D, n_known) depois transposto em dot
    S = np.zeros((len(face_embs), len(known_embs)), dtype=np.float32)
    for i, q in enumerate(face_embs):
        if q is None:
            continue
        qn = l2_normalize(q).astype(np.float32)
        S[i, :] = (qn @ K).astype(np.float32)  # produto interno (cosseno) pois L2-normalizado
    return S

def greedy_global_assignment(S: np.ndarray) -> list[tuple[int, int, float]]:
    """
    Matching guloso global: ordena pares por similaridade desc e aloca evitando reutilização
    de face/motorista. Retorna lista de (face_idx, driver_idx, sim).
    """
    if S.size == 0:
        return []
    pairs = [ (float(S[i,j]), i, j) for i in range(S.shape[0]) for j in range(S.shape[1]) ]
    pairs.sort(key=lambda t: t[0], reverse=True)
    used_faces, used_drivers, match = set(), set(), []
    for sim, i, j in pairs:
        if i in used_faces or j in used_drivers:
            continue
        # só marca; decisão final (threshold/margem) vem depois
        used_faces.add(i); used_drivers.add(j); match.append((i, j, float(sim)))
    return match

def check_enrollment_conflicts(known_embs: list[np.ndarray], id_name_pairs: list[tuple[str,str]], warn_threshold: float = 0.72) -> list[str]:
    """
    Detecta pares de motoristas cadastrados muito semelhantes (risco de confusão).
    """
    warnings: list[str] = []
    n = len(known_embs)
    if n < 2:
        return warnings
    embs = [l2_normalize(e) for e in known_embs]
    for a in range(n):
        for b in range(a+1, n):
            sim = float(np.dot(embs[a], embs[b]))
            if sim >= warn_threshold:
                wa = f"Possível conflito entre '{id_name_pairs[a][1]}' e '{id_name_pairs[b][1]}' (sim={sim:.3f})."
                warnings.append(wa)
    return warnings

# Garante que todas as funções acima existem antes de iniciar o servidor
if __name__ == "__main__":
    main()
