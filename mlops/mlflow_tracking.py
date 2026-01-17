"""
MLflow tracking pour le suivi des expériences DeblurGAN (version stable locale)
"""

import mlflow
import numpy as np
from typing import Dict, Optional
from datetime import datetime


class DeblurGANTracker:
    def __init__(self, experiment_name="DeblurGAN-Forensic-INPT", tracking_uri="./mlruns"):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        print(f"✅ MLflow experiment: {experiment_name}")
        print(f"📊 Tracking URI: {tracking_uri}")

    def start_run(self, run_name=None):
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow.start_run(run_name=run_name)
        print(f"🚀 Started run: {run_name}")
        return mlflow.active_run().info.run_id

    def log_params(self, params: Dict):
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict, step=None):
        for k, v in metrics.items():
            mlflow.log_metric(k, v, step=step)

    def log_training_metrics(self, epoch, **metrics):
        self.log_metrics(metrics, step=epoch)
        print(f"Epoch {epoch} | " + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        mlflow.log_artifact(local_path, artifact_path)

    def log_forensic_analysis(self, ela_score, autoencoder_score, tampering_score, confidence_level):
        mlflow.log_metrics({
            'forensic_ela_score': ela_score,
            'forensic_autoencoder_score': autoencoder_score,
            'forensic_tampering_score': tampering_score
        })
        mlflow.log_param('forensic_confidence_level', confidence_level)

    def log_baseline_comparison(self, wiener_psnr, wiener_ssim, gan_psnr, gan_ssim):
        mlflow.log_metrics({
            'baseline_wiener_psnr': wiener_psnr,
            'baseline_wiener_ssim': wiener_ssim,
            'deblurgan_psnr': gan_psnr,
            'deblurgan_ssim': gan_ssim,
            'improvement_psnr': gan_psnr - wiener_psnr,
            'improvement_ssim': gan_ssim - wiener_ssim
        })

    def log_dataset_info(self, dataset_name, train_size, val_size, test_size, image_shape):
        mlflow.log_params({
            'dataset_name': dataset_name,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'image_height': image_shape[0],
            'image_width': image_shape[1],
            'image_channels': image_shape[2]
        })

    def end_run(self, status="FINISHED"):
        mlflow.end_run(status=status)
        print(f"✅ Run ended with status: {status}")


class MetricsLogger:
    def __init__(self):
        self.history = {}

    def log(self, epoch, **metrics):
        if 'epoch' not in self.history:
            self.history['epoch'] = []
        self.history['epoch'].append(epoch)

        for k, v in metrics.items():
            self.history.setdefault(k, []).append(v)
