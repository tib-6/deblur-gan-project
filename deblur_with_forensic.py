"""
Pipeline complet : DeblurGAN + Forensic Analysis
"""

import cv2
import numpy as np
import os

from forensic.tampering_detector import TamperingDetector


def deblur_and_analyze(blurred_image_path: str, output_dir: str = "outputs"):
    print("=" * 70)
    print("🚀 PIPELINE : DeblurGAN + Forensic Analysis")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # ========== 1. CHARGEMENT ==========
    print("\n📂 Chargement de l'image floue...")
    blurred_img = cv2.imread(blurred_image_path)
    if blurred_img is None:
        raise ValueError(f"Impossible de charger : {blurred_image_path}")
    print(f"✅ Image chargée : {blurred_img.shape}")

    # ========== 2. DÉFLOUTAGE (placeholder pour l'instant) ==========
    print("\n🎨 Défloutage avec DeblurGAN...")
    print("⚠️ PLACEHOLDER : Filtre gaussien (remplacer plus tard par DeblurGAN)")
    deblurred_img = cv2.GaussianBlur(blurred_img, (5, 5), 0)
    print("✅ Défloutage terminé")

    # Sauvegarde
    deblurred_path = os.path.join(output_dir, "deblurred.png")
    cv2.imwrite(deblurred_path, deblurred_img)
    print(f"💾 Image défloutée sauvegardée : {deblurred_path}")

    # ========== 3. ANALYSE FORENSIQUE ==========
    print("\n🔬 Analyse forensique...")

    detector = TamperingDetector(
        ela_weight=1.0,
        autoencoder_weight=0.0,
        autoencoder_path=None
    )

    blurred_results = detector.analyze_image(blurred_img, include_details=True)
    deblurred_results = detector.analyze_image(deblurred_img, include_details=True)

    score_increase = deblurred_results["tampering_score"] - blurred_results["tampering_score"]

    # ========== 4. AFFICHAGE ==========
    print("\n" + "=" * 70)
    print("📊 RÉSULTATS DE L'ANALYSE")
    print("=" * 70)

    print("\n📷 IMAGE FLOUE :")
    print(f"   Score : {blurred_results['tampering_score']}/100")
    print(f"   Confiance : {blurred_results['confidence_level']}")

    print("\n🎨 IMAGE DÉFLOUTÉE :")
    print(f"   Score : {deblurred_results['tampering_score']}/100")
    print(f"   Confiance : {deblurred_results['confidence_level']}")

    print(f"\n📈 Variation du score : {score_increase:+.2f} points")

    verdict = "Aucune manipulation détectée" if deblurred_results["tampering_score"] < 40 else "Image suspecte"
    print(f"\n🎯 VERDICT : {verdict}")

    print("\n" + "=" * 70)

    # ========== 5. VISUALISATION ==========
    print("\n📸 Création de la visualisation comparative...")

    comparison_img = create_comparison_visualization(
        blurred_img,
        deblurred_img,
        blurred_results,
        deblurred_results
    )

    comparison_path = os.path.join(output_dir, "comparison.png")
    cv2.imwrite(comparison_path, comparison_img)
    print(f"💾 Visualisation sauvegardée : {comparison_path}")

    print("\n✅ Pipeline terminé avec succès!")

    return {
        "deblurred_image": deblurred_img,
        "forensic_results": {
            "blurred": blurred_results,
            "deblurred": deblurred_results,
            "score_increase": score_increase,
            "verdict": verdict
        },
        "output_paths": {
            "deblurred": deblurred_path,
            "comparison": comparison_path
        }
    }


def create_comparison_visualization(blurred, deblurred, blurred_results, deblurred_results):
    h, w = blurred.shape[:2]
    max_width = 800
    if w > max_width:
        scale = max_width / w
        new_h = int(h * scale)
        blurred = cv2.resize(blurred, (max_width, new_h))
        deblurred = cv2.resize(deblurred, (max_width, new_h))

    comparison = np.hstack([blurred, deblurred])
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(
        comparison,
        f"Floue - Score: {blurred_results['tampering_score']}",
        (10, 30),
        font,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        comparison,
        f"Defloute - Score: {deblurred_results['tampering_score']}",
        (comparison.shape[1] // 2 + 10, 30),
        font,
        1,
        (0, 255, 0),
        2
    )

    return comparison


def main():
    # 🔴 CHANGE ICI
    blurred_image_path = "input_images/sample1.jpg"

    if not os.path.exists(blurred_image_path):
        print(f"❌ ERREUR : Fichier non trouvé : {blurred_image_path}")
        return

    try:
        results = deblur_and_analyze(blurred_image_path)
        print("\n🎉 Succès ! Vérifie le dossier 'outputs/'")

    except Exception as e:
        print(f"\n❌ ERREUR : {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
