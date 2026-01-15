# Dans api/main.py, remplacer les placeholders par :

from deblurgan.model import Generator
from forensic.tampering_detector import TamperingDetector

@app.on_event("startup")
async def load_models():
    global generator_model, forensic_detector
    
    # Chargement du générateur DeblurGAN
    generator_model = Generator()
    generator_model.load_weights('models/deblurgan_weights.h5')
    
    # Chargement du détecteur forensique
    forensic_detector = TamperingDetector(
        autoencoder_path='models/autoencoder_weights.h5'
    )