"""
API FastAPI simplifiée pour DeblurGAN + Forensic
Version fonctionnelle sans dépendances lourdes
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import cv2
from io import BytesIO
import base64
import time
import sys
from pathlib import Path

# Ajoute le chemin du projet
sys.path.append(str(Path(__file__).resolve().parents[1]))

from forensic.tampering_detector import TamperingDetector

# ========== INITIALISATION FASTAPI ==========

app = FastAPI(
    title="DeblurGAN Forensic API",
    description="API pour défloutage et analyse forensique",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== VARIABLES GLOBALES ==========

forensic_detector = None
# deblurgan_model = None  # À ajouter plus tard


# ========== STARTUP ==========

@app.on_event("startup")
async def startup():
    """Chargement des modèles au démarrage"""
    global forensic_detector
    
    print("🚀 Démarrage de l'API...")
    
    # Initialisation du détecteur forensique
    forensic_detector = TamperingDetector(
        ela_weight=1.0,
        autoencoder_weight=0.0,  # Pas d'autoencoder pour l'instant
        autoencoder_path=None
    )
    
    print("✅ Forensic detector initialisé")
    
    # TODO : Charger DeblurGAN
    # global deblurgan_model
    # deblurgan_model = load_deblurgan_model()
    
    print("✅ API prête!")


# ========== ENDPOINTS ==========

@app.get("/")
async def root():
    """Endpoint racine"""
    return {
        "message": "DeblurGAN Forensic API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "deblur": "/deblur (POST)",
            "forensic": "/forensic (POST)",
            "compare": "/compare (POST)"
        }
    }


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "forensic_detector": forensic_detector is not None,
        "deblurgan_model": False  # À mettre True quand chargé
    }


@app.post("/deblur")
async def deblur_endpoint(file: UploadFile = File(...)):
    """
    Défloutage d'une image
    
    Usage:
        curl -X POST "http://localhost:8000/deblur" -F "file=@image.jpg"
    """
    try:
        start_time = time.time()
        
        # Lecture de l'image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Image invalide")
        
        # === DÉFLOUTAGE ===
        # TODO : Utiliser le vrai modèle DeblurGAN
        # deblurred = deblurgan_model.predict(image)
        
        # PLACEHOLDER : Filtre gaussien pour test
        deblurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Encodage
        _, buffer = cv2.imencode('.png', deblurred)
        
        processing_time = (time.time() - start_time) * 1000
        
        return StreamingResponse(
            BytesIO(buffer.tobytes()),
            media_type="image/png",
            headers={
                "X-Processing-Time": f"{processing_time:.2f}ms"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/forensic")
async def forensic_endpoint(file: UploadFile = File(...)):
    """
    Analyse forensique d'une image
    
    Usage:
        curl -X POST "http://localhost:8000/forensic" -F "file=@image.jpg"
    """
    try:
        # Lecture
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Image invalide")
        
        # Analyse
        results = forensic_detector.analyze_image(image, include_details=False)
        
        return JSONResponse(content={
            "success": True,
            "tampering_score": results['tampering_score'],
            "confidence_level": results['confidence_level'],
            "is_tampered": results['is_tampered'],
            "ela_score": results['ela_score'],
            "methods_used": results['methods_used']
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deblur-forensic")
async def deblur_with_forensic(file: UploadFile = File(...)):
    """
    Défloutage + Analyse forensique combinés
    
    Retourne l'image défloutée + le score forensique
    """
    try:
        start_time = time.time()
        
        # Lecture
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Image invalide")
        
        # Défloutage (placeholder)
        deblurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Analyse forensique
        forensic_results = forensic_detector.analyze_image(
            deblurred,
            include_details=False
        )
        
        # Encodage de l'image
        _, buffer = cv2.imencode('.png', deblurred)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        processing_time = (time.time() - start_time) * 1000
        
        return JSONResponse(content={
            "success": True,
            "deblurred_image_base64": img_base64,
            "processing_time_ms": processing_time,
            "forensic": {
                "tampering_score": forensic_results['tampering_score'],
                "confidence": forensic_results['confidence_level'],
                "is_tampered": forensic_results['is_tampered']
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare")
async def compare_images(
    blurred: UploadFile = File(...),
    deblurred: UploadFile = File(...)
):
    """
    Compare deux images (avant/après défloutage)
    
    Usage:
        curl -X POST "http://localhost:8000/compare" \
             -F "blurred=@blur.jpg" \
             -F "deblurred=@deblur.jpg"
    """
    try:
        # Lecture image floue
        blur_contents = await blurred.read()
        blur_arr = np.frombuffer(blur_contents, np.uint8)
        blur_img = cv2.imdecode(blur_arr, cv2.IMREAD_COLOR)
        
        # Lecture image défloutée
        deblur_contents = await deblurred.read()
        deblur_arr = np.frombuffer(deblur_contents, np.uint8)
        deblur_img = cv2.imdecode(deblur_arr, cv2.IMREAD_COLOR)
        
        if blur_img is None or deblur_img is None:
            raise HTTPException(status_code=400, detail="Images invalides")
        
        # Comparaison forensique
        comparison = forensic_detector.compare_before_after(
            blur_img,
            deblur_img
        )
        
        return JSONResponse(content=comparison)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== LANCEMENT ==========

if __name__ == "__main__":
    import uvicorn
    
    print("="*70)
    print("🚀 DÉMARRAGE DE L'API FASTAPI")
    print("="*70)
    print("📍 URL: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    print("="*70)
    
    uvicorn.run(
        "main_simple:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )