"""
Error Level Analysis (ELA) pour détecter les manipulations d'image
Principe : Les zones manipulées ont des niveaux d'erreur JPEG différents
"""

import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from typing import Tuple, Optional
import matplotlib.pyplot as plt


class ELADetector:
    """
    Détecte les artefacts de manipulation via Error Level Analysis
    """
    
    def __init__(self, quality: int = 90, scale: int = 15):
        """
        Args:
            quality: Qualité JPEG pour la recompression (90 par défaut)
            scale: Facteur d'amplification des différences (15 par défaut)
        """
        self.quality = quality
        self.scale = scale
    
    def compute_ela(self, image: np.ndarray) -> np.ndarray:
        """
        Calcule l'Error Level Analysis d'une image
        
        Args:
            image: Image en format numpy (H, W, 3) BGR ou RGB
            
        Returns:
            ELA map: Carte des erreurs (valeurs élevées = zones suspectes)
        """
        # Conversion BGR -> RGB si nécessaire
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Conversion en PIL Image
        pil_image = Image.fromarray(image_rgb.astype('uint8'))
        
        # Recompression JPEG en mémoire
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=self.quality)
        buffer.seek(0)
        
        # Rechargement de l'image compressée
        compressed_image = Image.open(buffer)
        compressed_array = np.array(compressed_image)
        
        # Calcul de la différence absolue
        ela_image = np.abs(image_rgb.astype('float32') - compressed_array.astype('float32'))
        
        # Amplification des différences
        ela_image = ela_image * self.scale
        ela_image = np.clip(ela_image, 0, 255).astype('uint8')
        
        # Conversion en niveaux de gris
        ela_gray = cv2.cvtColor(ela_image, cv2.COLOR_RGB2GRAY)
        
        return ela_gray
    
    def compute_ela_score(self, image: np.ndarray) -> float:
        """
        Calcule un score global de manipulation (0-1)
        Score élevé = forte probabilité de manipulation
        
        Args:
            image: Image en format numpy
            
        Returns:
            score: Valeur entre 0 et 1
        """
        ela_map = self.compute_ela(image)
        
        # Normalisation et calcul de la variance
        ela_normalized = ela_map.astype('float32') / 255.0
        
        # Score basé sur la variance et la moyenne
        mean_error = np.mean(ela_normalized)
        std_error = np.std(ela_normalized)
        
        # Score combiné (heuristique)
        score = min(1.0, (mean_error * 2 + std_error) / 2)
        
        return float(score)
    
    def detect_suspicious_regions(
        self, 
        image: np.ndarray, 
        threshold: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Détecte les régions suspectes via seuillage adaptatif
        
        Args:
            image: Image originale
            threshold: Seuil de détection (0-1)
            
        Returns:
            mask: Masque binaire des zones suspectes
            heatmap: Carte de chaleur colorée
        """
        ela_map = self.compute_ela(image)
        
        # Seuillage adaptatif
        threshold_value = int(threshold * 255)
        _, mask = cv2.threshold(ela_map, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Filtrage morphologique (enlever le bruit)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Création de la heatmap colorée
        heatmap = cv2.applyColorMap(ela_map, cv2.COLORMAP_JET)
        
        return mask, heatmap
    
    def visualize_analysis(
        self, 
        original: np.ndarray, 
        deblurred: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualise l'analyse forensique comparative
        
        Args:
            original: Image floue originale
            deblurred: Image défloutée
            save_path: Chemin de sauvegarde (optionnel)
        """
        # Calcul ELA pour les deux images
        ela_original = self.compute_ela(original)
        ela_deblurred = self.compute_ela(deblurred)
        
        # Scores
        score_original = self.compute_ela_score(original)
        score_deblurred = self.compute_ela_score(deblurred)
        
        # Visualisation
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Ligne 1 : Image originale
        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Blurred Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(ela_original, cmap='jet')
        axes[0, 1].set_title(f'ELA Map (Score: {score_original:.3f})')
        axes[0, 1].axis('off')
        
        mask_orig, _ = self.detect_suspicious_regions(original)
        axes[0, 2].imshow(mask_orig, cmap='gray')
        axes[0, 2].set_title('Suspicious Regions')
        axes[0, 2].axis('off')
        
        # Ligne 2 : Image défloutée
        axes[1, 0].imshow(cv2.cvtColor(deblurred, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Deblurred Image')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(ela_deblurred, cmap='jet')
        axes[1, 1].set_title(f'ELA Map (Score: {score_deblurred:.3f})')
        axes[1, 1].axis('off')
        
        mask_deblur, _ = self.detect_suspicious_regions(deblurred)
        axes[1, 2].imshow(mask_deblur, cmap='gray')
        axes[1, 2].set_title('Suspicious Regions')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Analysis saved to {save_path}")
        
        plt.show()


# Fonction utilitaire pour usage rapide
def quick_ela_check(image_path: str) -> dict:
    """
    Analyse ELA rapide d'une image
    
    Args:
        image_path: Chemin vers l'image
        
    Returns:
        Dictionnaire avec score et métadonnées
    """
    detector = ELADetector()
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    score = detector.compute_ela_score(image)
    ela_map = detector.compute_ela(image)
    
    return {
        'score': score,
        'is_suspicious': score > 0.4,  # Seuil empirique
        'mean_error': np.mean(ela_map),
        'max_error': np.max(ela_map),
        'image_shape': image.shape
    }


if __name__ == "__main__":
    # Test basique
    print("🔍 ELA Detector ready!")
    print("Usage: detector = ELADetector()")
    print("       score = detector.compute_ela_score(image)")