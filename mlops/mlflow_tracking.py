"""
MLflow tracking pour le suivi des expériences DeblurGAN
"""

import mlflow
import mlflow.keras
import numpy as np
from typing import Dict, Optional
import json
from datetime import datetime
from pathlib import Path


class DeblurGANTracker:
    """
    Gestionnaire de tracking MLflow pour DeblurGAN
    """
    
    def __init__(
        self,
        experiment_name: str = "DeblurGAN-Forensic",
        tracking_uri: str = "./mlruns"
    ):
        """
        Args:
            experiment_name: Nom de l'expérience MLflow
            tracking_uri: URI du serveur de tracking
        """
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        
        # Création ou récupération de l'expérience
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(experiment_name)
        print(f"✅ MLflow experiment: {experiment_name}")
        print(f"📊 Tracking URI: {tracking_uri}")
    
    def start_run(self, run_name: Optional[str] = None) -> str:
        """
        Démarre une nouvelle run MLflow
        
        Args:
            run_name: Nom de la run (optionnel)
            
        Returns:
            Run ID
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        mlflow.start_run(run_name=run_name)
        run_id = mlflow.active_run().info.run_id
        print(f"🚀 Started run: {run_name} (ID: {run_id})")
        
        return run_id
    
    def log_params(self, params: Dict) -> None:
        """
        Log des hyperparamètres
        
        Args:
            params: Dictionnaire d'hyperparamètres
        """
        for key, value in params.items():
            mlflow.log_param(key, value)
        print(f"📝 Logged {len(params)} parameters")
    
    def log_metrics(
        self,
        metrics: Dict,
        step: Optional[int] = None
    ) -> None:
        """
        Log des métriques
        
        Args:
            metrics: Dictionnaire de métriques
            step: Numéro d'époque (optionnel)
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_training_metrics(
        self,
        epoch: int,
        generator_loss: float,
        discriminator_loss: float,
        perceptual_loss: float,
        psnr: float,
        ssim: float,
        forensic_score: Optional[float] = None
    ) -> None:
        """
        Log des métriques d'entraînement
        
        Args:
            epoch: Numéro d'époque
            generator_loss: Loss du générateur
            discriminator_loss: Loss du discriminateur
            perceptual_loss: Perceptual loss
            psnr: Peak Signal-to-Noise Ratio
            ssim: Structural Similarity Index
            forensic_score: Score forensique (optionnel)
        """
        metrics = {
            'generator_loss': generator_loss,
            'discriminator_loss': discriminator_loss,
            'perceptual_loss': perceptual_loss,
            'psnr': psnr,
            'ssim': ssim
        }
        
        if forensic_score is not None:
            metrics['forensic_score'] = forensic_score
        
        self.log_metrics(metrics, step=epoch)
        
        # Logging console
        print(f"Epoch {epoch:03d} | "
              f"G_loss: {generator_loss:.4f} | "
              f"D_loss: {discriminator_loss:.4f} | "
              f"PSNR: {psnr:.2f} | "
              f"SSIM: {ssim:.4f}")
    
    def log_model(
        self,
        model,
        artifact_path: str = "models",
        registered_model_name: Optional[str] = None
    ) -> None:
        """
        Sauvegarde le modèle dans MLflow
        
        Args:
            model: Modèle Keras à sauvegarder
            artifact_path: Chemin de l'artifact
            registered_model_name: Nom du modèle dans le registry
        """
        mlflow.keras.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name
        )
        print(f"💾 Model logged to MLflow")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log un fichier artifact
        
        Args:
            local_path: Chemin local du fichier
            artifact_path: Chemin dans MLflow (optionnel)
        """
        mlflow.log_artifact(local_path, artifact_path)
        print(f"📦 Artifact logged: {local_path}")
    
    def log_images(
        self,
        blurred: np.ndarray,
        deblurred: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        epoch: Optional[int] = None
    ) -> None:
        """
        Log des exemples d'images
        
        Args:
            blurred: Image floue
            deblurred: Image défloutée
            ground_truth: Image nette de référence
            epoch: Numéro d'époque
        """
        import cv2
        from tempfile import NamedTemporaryFile
        
        # Création d'une comparaison visuelle
        if ground_truth is not None:
            comparison = np.hstack([blurred, deblurred, ground_truth])
        else:
            comparison = np.hstack([blurred, deblurred])
        
        # Sauvegarde temporaire
        with NamedTemporaryFile(suffix='.png', delete=False) as f:
            cv2.imwrite(f.name, comparison)
            
            # Log dans MLflow
            if epoch is not None:
                mlflow.log_artifact(f.name, f"images/epoch_{epoch:03d}.png")
            else:
                mlflow.log_artifact(f.name, "images/comparison.png")
    
    def log_forensic_analysis(
        self,
        ela_score: float,
        autoencoder_score: float,
        tampering_score: int,
        confidence_level: str
    ) -> None:
        """
        Log d'une analyse forensique
        
        Args:
            ela_score: Score ELA
            autoencoder_score: Score autoencoder
            tampering_score: Score de manipulation
            confidence_level: Niveau de confiance
        """
        mlflow.log_metrics({
            'forensic_ela_score': ela_score,
            'forensic_autoencoder_score': autoencoder_score,
            'forensic_tampering_score': tampering_score
        })
        
        mlflow.log_param('forensic_confidence_level', confidence_level)
    
    def log_baseline_comparison(
        self,
        wiener_psnr: float,
        wiener_ssim: float,
        gan_psnr: float,
        gan_ssim: float
    ) -> None:
        """
        Log de la comparaison avec baseline
        
        Args:
            wiener_psnr: PSNR du filtre de Wiener
            wiener_ssim: SSIM du filtre de Wiener
            gan_psnr: PSNR de DeblurGAN
            gan_ssim: SSIM de DeblurGAN
        """
        mlflow.log_metrics({
            'baseline_wiener_psnr': wiener_psnr,
            'baseline_wiener_ssim': wiener_ssim,
            'deblurgan_psnr': gan_psnr,
            'deblurgan_ssim': gan_ssim,
            'improvement_psnr': gan_psnr - wiener_psnr,
            'improvement_ssim': gan_ssim - wiener_ssim
        })
        
        print(f"📊 Baseline comparison logged")
        print(f"   Wiener: PSNR={wiener_psnr:.2f}, SSIM={wiener_ssim:.4f}")
        print(f"   GAN:    PSNR={gan_psnr:.2f}, SSIM={gan_ssim:.4f}")
        print(f"   Gain:   PSNR={gan_psnr-wiener_psnr:+.2f}, SSIM={gan_ssim-wiener_ssim:+.4f}")
    
    def log_dataset_info(
        self,
        dataset_name: str,
        train_size: int,
        val_size: int,
        test_size: int,
        image_shape: tuple
    ) -> None:
        """
        Log des informations sur le dataset
        
        Args:
            dataset_name: Nom du dataset
            train_size: Taille du training set
            val_size: Taille du validation set
            test_size: Taille du test set
            image_shape: Dimensions des images
        """
        mlflow.log_params({
            'dataset_name': dataset_name,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'image_height': image_shape[0],
            'image_width': image_shape[1],
            'image_channels': image_shape[2]
        })
    
    def end_run(self, status: str = "FINISHED") -> None:
        """
        Termine la run courante
        
        Args:
            status: Statut final (FINISHED, FAILED, KILLED)
        """
        mlflow.end_run(status=status)
        print(f"✅ Run ended with status: {status}")
    
    def load_best_model(
        self,
        metric: str = "psnr",
        ascending: bool = False
    ) -> str:
        """
        Charge le meilleur modèle selon une métrique
        
        Args:
            metric: Métrique de sélection
            ascending: Tri ascendant ou descendant
            
        Returns:
            Run ID du meilleur modèle
        """
        from mlflow.tracking import MlflowClient
        
        client = MlflowClient()
        runs = client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"]
        )
        
        if not runs:
            raise ValueError("No runs found!")
        
        best_run = runs[0]
        print(f"🏆 Best model (by {metric}): Run ID {best_run.info.run_id}")
        print(f"   {metric}: {best_run.data.metrics.get(metric, 'N/A')}")
        
        return best_run.info.run_id


class MetricsLogger:
    """
    Logger léger pour suivre les métriques pendant l'entraînement
    """
    
    def __init__(self):
        self.history = {
            'epoch': [],
            'generator_loss': [],
            'discriminator_loss': [],
            'psnr': [],
            'ssim': [],
            'forensic_score': []
        }
    
    def log(self, epoch: int, **metrics):
        """Log des métriques"""
        self.history['epoch'].append(epoch)
        
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def save(self, path: str):
        """Sauvegarde l'historique"""
        import json
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"📊 Metrics saved to {path}")
    
    def plot(self, save_path: Optional[str] = None):
        """Visualise les métriques"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Generator loss
        axes[0, 0].plot(self.history['epoch'], self.history['generator_loss'])
        axes[0, 0].set_title('Generator Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].grid(True)
        
        # Discriminator loss
        axes[0, 1].plot(self.history['epoch'], self.history['discriminator_loss'])
        axes[0, 1].set_title('Discriminator Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].grid(True)
        
        # PSNR
        axes[1, 0].plot(self.history['epoch'], self.history['psnr'])
        axes[1, 0].set_title('PSNR')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].grid(True)
        
        # SSIM
        axes[1, 1].plot(self.history['epoch'], self.history['ssim'])
        axes[1, 1].set_title('SSIM')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"📈 Plot saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    print("📊 MLflow Tracker ready!")
    print("Usage: tracker = DeblurGANTracker()")
    print("       tracker.start_run('experiment_1')")