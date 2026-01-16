"""
Test simple du module forensic
"""

import cv2
import numpy as np
from forensic.ela import ELADetector
from forensic.tampering_detector import TamperingDetector

def test_ela():
    """Test de l'ELA sur une image"""
    print(" Test ELA Detector...")
    
    # Charge une image défloutée (résultat de DeblurGAN)
    image_path = "input_images/sample1.jpg"  
    image = cv2.imread(image_path)
    
    if image is None:
        print(" Erreur : Image non trouvée")
        return
    
    # Initialisation du détecteur
    detector = ELADetector(quality=90, scale=15)
    
    # Calcul du score
    score = detector.compute_ela_score(image)
    print(f" ELA Score: {score:.4f}")
    
    # Visualisation
    detector.visualize_analysis(
        original=image,
        deblurred=image,
        save_path="outputs/ela_analysis.png"
    )
    
    print(" Analyse sauvegardée dans outputs/ela_analysis.png")


def test_tampering_simple():
    """Test du détecteur de manipulation (sans autoencoder)"""
    print("\nTest Tampering Detector...")
    
    # Charge une image
    image_path = "output_images/sample1.jpg"  
    image = cv2.imread(image_path)
    
    if image is None:
        print(" Erreur : Image non trouvée")
        return
    
    # Initialisation (sans autoencoder pour l'instant)
    detector = TamperingDetector(
        ela_weight=1.0,  # 100% ELA (pas d'autoencoder)
        autoencoder_weight=0.0,
        autoencoder_path=None
    )
    
    # Analyse
    results = detector.analyze_image(image, include_details=True)
    
    # Affichage
    print(f" Tampering Score: {results['tampering_score']}/100")
    print(f"   Confidence: {results['confidence_level']}")
    print(f"   Is Tampered: {results['is_tampered']}")
    print(f"   ELA Score: {results['ela_score']:.4f}")


if __name__ == "__main__":
    print("="*60)
    print("🧪 TEST DU MODULE FORENSIC")
    print("="*60)
    
    # Crée le dossier outputs s'il n'existe pas
    import os
    os.makedirs("outputs", exist_ok=True)
    
    # Tests
    test_ela()
    test_tampering_simple()
    
    print("\n" + "="*60)
    print("Tests terminés avec succès!")
    print("="*60)