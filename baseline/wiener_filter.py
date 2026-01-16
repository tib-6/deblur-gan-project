"""
Filtre de Wiener pour défloutage classique (baseline de comparaison)
"""

import numpy as np
import cv2
from scipy.signal import convolve2d
from typing import Tuple, Optional


class WienerFilter:
    """
    Implémentation du filtre de Wiener pour le défloutage d'images
    """
    
    def __init__(self, K: float = 0.01):
        """
        Args:
            K: Paramètre de régularisation (rapport signal/bruit)
        """
        self.K = K
    
    def estimate_blur_kernel(
        self, 
        kernel_size: int = 15,
        angle: Optional[float] = None,
        length: Optional[float] = None
    ) -> np.ndarray:
        """
        Estime un noyau de flou de mouvement linéaire
        
        Args:
            kernel_size: Taille du noyau
            angle: Angle du mouvement (degrés)
            length: Longueur du mouvement
            
        Returns:
            Noyau de flou estimé
        """
        if angle is None:
            angle = 0
        if length is None:
            length = kernel_size // 2
        
        # Création d'un noyau vide
        kernel = np.zeros((kernel_size, kernel_size))
        
        # Centre du noyau
        center = kernel_size // 2
        
        # Conversion de l'angle en radians
        angle_rad = np.deg2rad(angle)
        
        # Génération du mouvement linéaire
        for i in range(-int(length), int(length)):
            x = int(center + i * np.cos(angle_rad))
            y = int(center + i * np.sin(angle_rad))
            
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1
        
        # Normalisation
        if kernel.sum() > 0:
            kernel /= kernel.sum()
        else:
            kernel[center, center] = 1
        
        return kernel
    
    def wiener_filter_frequency(
        self,
        image: np.ndarray,
        psf: np.ndarray,
        K: Optional[float] = None
    ) -> np.ndarray:
        """
        Applique le filtre de Wiener dans le domaine fréquentiel
        
        Args:
            image: Image floue (2D grayscale)
            psf: Point Spread Function (noyau de flou)
            K: Paramètre de régularisation (optionnel)
            
        Returns:
            Image défloutée
        """
        if K is None:
            K = self.K
        
        # Padding pour éviter les effets de bord
        image_padded = self._pad_image(image, psf.shape)
        
        # FFT de l'image et du PSF
        image_fft = np.fft.fft2(image_padded)
        psf_padded = self._pad_psf(psf, image_padded.shape)
        psf_fft = np.fft.fft2(psf_padded)
        
        # Calcul du filtre de Wiener
        psf_conj = np.conj(psf_fft)
        wiener_filter = psf_conj / (np.abs(psf_fft)**2 + K)
        
        # Application du filtre
        restored_fft = image_fft * wiener_filter
        restored = np.fft.ifft2(restored_fft)
        restored = np.real(restored)
        
        # Crop au dimensions originales
        restored = self._crop_to_original(restored, image.shape)
        
        # Normalisation
        restored = np.clip(restored, 0, 255)
        
        return restored.astype(np.uint8)
    
    def deblur_image(
        self,
        image: np.ndarray,
        kernel_size: int = 15,
        angle: float = 0,
        length: float = 10,
        K: Optional[float] = None
    ) -> np.ndarray:
        """
        Défloutage complet d'une image couleur
        
        Args:
            image: Image floue BGR (H, W, 3)
            kernel_size: Taille du noyau
            angle: Angle du mouvement
            length: Longueur du mouvement
            K: Paramètre de régularisation
            
        Returns:
            Image défloutée BGR
        """
        if K is None:
            K = self.K
        
        # Estimation du noyau de flou
        psf = self.estimate_blur_kernel(kernel_size, angle, length)
        
        # Traitement par canal
        deblurred_channels = []
        
        for i in range(3):  # BGR
            channel = image[:, :, i]
            deblurred_channel = self.wiener_filter_frequency(channel, psf, K)
            deblurred_channels.append(deblurred_channel)
        
        # Reconstruction de l'image
        deblurred_image = np.stack(deblurred_channels, axis=-1)
        
        return deblurred_image.astype(np.uint8)
    
    def blind_deblur(
        self,
        image: np.ndarray,
        kernel_sizes: list = [11, 15, 21],
        angles: list = [0, 45, 90, 135],
        K: Optional[float] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Défloutage aveugle avec recherche de paramètres optimaux
        
        Args:
            image: Image floue
            kernel_sizes: Liste de tailles à tester
            angles: Liste d'angles à tester
            K: Paramètre de régularisation
            
        Returns:
            Meilleure image défloutée et paramètres
        """
        best_score = float('-inf')
        best_result = None
        best_params = {}
        
        for ks in kernel_sizes:
            for angle in angles:
                # Défloutage
                deblurred = self.deblur_image(
                    image, 
                    kernel_size=ks,
                    angle=angle,
                    K=K
                )
                
                # Évaluation (gradient moyen comme métrique de netteté)
                score = self._compute_sharpness(deblurred)
                
                if score > best_score:
                    best_score = score
                    best_result = deblurred
                    best_params = {
                        'kernel_size': ks,
                        'angle': angle,
                        'sharpness_score': score
                    }
        
        return best_result, best_params
    
    def _compute_sharpness(self, image: np.ndarray) -> float:
        """
        Calcule un score de netteté (variance du Laplacien)
        
        Args:
            image: Image à évaluer
            
        Returns:
            Score de netteté
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        score = laplacian.var()
        return float(score)
    
    def _pad_image(self, image: np.ndarray, psf_shape: Tuple[int, int]) -> np.ndarray:
        """Padding de l'image"""
        pad_h = psf_shape[0] // 2
        pad_w = psf_shape[1] // 2
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    
    def _pad_psf(self, psf: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Padding du PSF"""
        padded = np.zeros(target_shape)
        h, w = psf.shape
        padded[:h, :w] = psf
        return np.roll(np.roll(padded, -h//2, axis=0), -w//2, axis=1)
    
    def _crop_to_original(self, image: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """Crop aux dimensions originales"""
        return image[:original_shape[0], :original_shape[1]]


def compare_wiener_gan(
    blurred_image: np.ndarray,
    gan_result: np.ndarray,
    ground_truth: Optional[np.ndarray] = None
) -> dict:
    """
    Compare le filtre de Wiener avec DeblurGAN
    
    Args:
        blurred_image: Image floue
        gan_result: Résultat DeblurGAN
        ground_truth: Image nette de référence (optionnel)
        
    Returns:
        Dictionnaire de comparaison
    """
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    
    # Défloutage Wiener
    wiener = WienerFilter()
    wiener_result, params = wiener.blind_deblur(blurred_image)
    
    results = {
        'wiener_params': params,
        'wiener_sharpness': wiener._compute_sharpness(wiener_result),
        'gan_sharpness': wiener._compute_sharpness(gan_result)
    }
    
    # Si on a une ground truth
    if ground_truth is not None:
        # PSNR
        wiener_psnr = peak_signal_noise_ratio(ground_truth, wiener_result)
        gan_psnr = peak_signal_noise_ratio(ground_truth, gan_result)
        
        # SSIM
        wiener_ssim = structural_similarity(
            cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(wiener_result, cv2.COLOR_BGR2GRAY)
        )
        gan_ssim = structural_similarity(
            cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(gan_result, cv2.COLOR_BGR2GRAY)
        )
        
        results['wiener_psnr'] = wiener_psnr
        results['gan_psnr'] = gan_psnr
        results['wiener_ssim'] = wiener_ssim
        results['gan_ssim'] = gan_ssim
        
        results['gan_improvement_psnr'] = gan_psnr - wiener_psnr
        results['gan_improvement_ssim'] = gan_ssim - wiener_ssim
    
    return results


if __name__ == "__main__":
    print("🔧 Wiener Filter ready!")
    print("Usage: wiener = WienerFilter()")
    print("       deblurred = wiener.deblur_image(image)")