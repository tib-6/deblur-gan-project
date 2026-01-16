"""
Autoencoder pour la détection d'anomalies dans les images défloutées
Principe : Entraîné sur images nettes, il détecte les artefacts inhabituels
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, Optional
import cv2
import matplotlib.pyplot as plt


class AnomalyAutoencoder:
    """
    Autoencoder pour détecter les artefacts de défloutage anormaux
    """
    
    def __init__(
        self, 
        input_shape: Tuple[int, int, int] = (256, 256, 3),
        latent_dim: int = 128
    ):
        """
        Args:
            input_shape: Dimensions des images (H, W, C)
            latent_dim: Dimension de l'espace latent
        """
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.model = None
        self.encoder = None
        self.decoder = None
        self.threshold = None  # Sera calculé après entraînement
        
    def build_model(self) -> Model:
        """
        Construit un autoencoder convolutif profond
        
        Returns:
            model: Modèle Keras compilé
        """
        # ========== ENCODER ==========
        encoder_input = layers.Input(shape=self.input_shape, name='input')
        
        # Block 1
        x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(encoder_input)
        x = layers.BatchNormalization()(x)
        
        # Block 2
        x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Block 3
        x = layers.Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Block 4
        x = layers.Conv2D(512, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Latent space
        x = layers.Flatten()(x)
        latent = layers.Dense(self.latent_dim, activation='relu', name='latent')(x)
        
        self.encoder = Model(encoder_input, latent, name='encoder')
        
        # ========== DECODER ==========
        latent_input = layers.Input(shape=(self.latent_dim,), name='latent_input')
        
        # Reshape to spatial dimensions
        h, w = self.input_shape[0] // 16, self.input_shape[1] // 16
        x = layers.Dense(h * w * 512, activation='relu')(latent_input)
        x = layers.Reshape((h, w, 512))(x)
        
        # Block 1
        x = layers.Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Block 2
        x = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Block 3
        x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Block 4
        x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Output
        decoder_output = layers.Conv2D(3, 3, padding='same', activation='sigmoid')(x)
        
        self.decoder = Model(latent_input, decoder_output, name='decoder')
        
        # ========== FULL AUTOENCODER ==========
        autoencoder_output = self.decoder(self.encoder(encoder_input))
        self.model = Model(encoder_input, autoencoder_output, name='autoencoder')
        
        # Compilation
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def train(
        self, 
        train_images: np.ndarray,
        validation_images: np.ndarray,
        epochs: int = 50,
        batch_size: int = 16
    ) -> keras.callbacks.History:
        """
        Entraîne l'autoencoder sur des images NETTES uniquement
        
        Args:
            train_images: Images d'entraînement (nettes)
            validation_images: Images de validation (nettes)
            epochs: Nombre d'époques
            batch_size: Taille des batchs
            
        Returns:
            history: Historique d'entraînement
        """
        if self.model is None:
            self.build_model()
        
        # Normalisation
        train_images = train_images.astype('float32') / 255.0
        validation_images = validation_images.astype('float32') / 255.0
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Entraînement (reconstruction de soi-même)
        history = self.model.fit(
            train_images, train_images,
            validation_data=(validation_images, validation_images),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Calcul du seuil d'anomalie sur la validation
        self._compute_threshold(validation_images)
        
        return history
    
    def _compute_threshold(self, validation_images: np.ndarray) -> None:
        """
        Calcule le seuil de détection d'anomalies (percentile 95)
        
        Args:
            validation_images: Images de validation (nettes)
        """
        # Reconstruction
        reconstructed = self.model.predict(validation_images, verbose=0)
        
        # Erreurs de reconstruction
        reconstruction_errors = np.mean(
            np.square(validation_images - reconstructed),
            axis=(1, 2, 3)
        )
        
        # Seuil au 95e percentile
        self.threshold = np.percentile(reconstruction_errors, 95)
        print(f"✅ Anomaly threshold set to: {self.threshold:.6f}")
    
    def detect_anomaly(
        self, 
        image: np.ndarray,
        return_details: bool = False
    ) -> dict:
        """
        Détecte si une image contient des anomalies
        
        Args:
            image: Image à analyser (H, W, 3)
            return_details: Retourner les détails (reconstruction, error map)
            
        Returns:
            Dictionnaire avec score et métadonnées
        """
        if self.model is None:
            raise ValueError("Model not trained! Call train() first.")
        
        # Normalisation
        image_norm = image.astype('float32') / 255.0
        image_batch = np.expand_dims(image_norm, axis=0)
        
        # Reconstruction
        reconstructed = self.model.predict(image_batch, verbose=0)[0]
        
        # Erreur de reconstruction
        reconstruction_error = np.mean(np.square(image_norm - reconstructed))
        
        # Carte d'erreur pixel par pixel
        error_map = np.mean(np.square(image_norm - reconstructed), axis=-1)
        
        # Détection d'anomalie
        is_anomaly = reconstruction_error > self.threshold if self.threshold else False
        
        result = {
            'reconstruction_error': float(reconstruction_error),
            'is_anomaly': bool(is_anomaly),
            'threshold': float(self.threshold) if self.threshold else None,
            'anomaly_score': float(reconstruction_error / (self.threshold + 1e-8)) if self.threshold else None
        }
        
        if return_details:
            result['reconstructed_image'] = (reconstructed * 255).astype('uint8')
            result['error_map'] = error_map
        
        return result
    
    def visualize_anomaly(
        self,
        image: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualise l'analyse d'anomalie
        
        Args:
            image: Image à analyser
            save_path: Chemin de sauvegarde
        """
        result = self.detect_anomaly(image, return_details=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Image originale
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Image reconstruite
        axes[1].imshow(result['reconstructed_image'])
        axes[1].set_title(f"Reconstructed (Error: {result['reconstruction_error']:.4f})")
        axes[1].axis('off')
        
        # Carte d'erreur
        im = axes[2].imshow(result['error_map'], cmap='hot')
        axes[2].set_title(f"Error Map (Anomaly: {result['is_anomaly']})")
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved to {save_path}")
        
        plt.show()
    
    def save_model(self, path: str) -> None:
        """Sauvegarde le modèle"""
        if self.model is None:
            raise ValueError("No model to save!")
        self.model.save(path)
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Charge un modèle pré-entraîné"""
        self.model = keras.models.load_model(path)
        print(f"✅ Model loaded from {path}")


if __name__ == "__main__":
    print("🤖 Anomaly Autoencoder ready!")
    print("Usage: model = AnomalyAutoencoder()")
    print("       model.build_model()")
    print("       model.train(sharp_images, val_images)")