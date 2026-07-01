import os
import cv2
import numpy as np
from loguru import logger

class FaceNotDetectedError(Exception):
    """Ngoại lệ khi không phát hiện được khuôn mặt trong ảnh tham chiếu."""
    pass

_face_app = None

def _get_face_app(device: str = "cuda"):
    global _face_app
    if _face_app is None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError("Vui lòng cài đặt insightface: pip install insightface onnxruntime-gpu")
            
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        
        logger.info(f"Loading InsightFace buffalo_l on {device}")
        _face_app = FaceAnalysis(name='buffalo_l', providers=providers)
        _face_app.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(640, 640))
        
    return _face_app

def extract_and_save_face_embedding(
    image_path: str,
    output_path: str,
    device: str = "cuda"
) -> np.ndarray:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
        
    app = _get_face_app(device)
    faces = app.get(img)
    
    if not faces:
        raise FaceNotDetectedError(
            f"Không tìm thấy khuôn mặt trong ảnh: {image_path}. "
            "Vui lòng chọn ảnh rõ khuôn mặt, không bị che khuất."
        )
        
    if len(faces) > 1:
        logger.info(f"Tìm thấy {len(faces)} khuôn mặt. Đang chọn khuôn mặt lớn nhất...")
        faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
        
    face = faces[0]
    embedding = face.normed_embedding
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embedding)
    logger.info(f"Saved face embedding to {output_path}")
    
    return embedding

def load_face_embedding(embedding_path: str) -> np.ndarray:
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embedding not found: {embedding_path}")
    return np.load(embedding_path)
