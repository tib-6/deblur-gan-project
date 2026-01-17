"""
Interface Streamlit pour DeblurGAN + Forensic Analysis
Version simplifiée et fonctionnelle
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import sys
from pathlib import Path
import os

# Ajoute le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))

from forensic.tampering_detector import TamperingDetector

# ========== CONFIGURATION ==========
st.set_page_config(
    page_title="DeblurGAN Forensic",
    page_icon="🔍",
    layout="wide"
)

# ========== STYLES CSS ==========
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ========== INITIALISATION ==========
@st.cache_resource
def load_detector():
    """Charge le détecteur forensique (une seule fois)"""
    return TamperingDetector(
        ela_weight=1.0,
        autoencoder_weight=0.0,
        autoencoder_path=None
    )

detector = load_detector()

# ========== FONCTIONS UTILITAIRES ==========

def deblur_image(image):
    """
    Défloutage d'image
    TODO: Remplacer par le vrai modèle DeblurGAN
    """
    # PLACEHOLDER : Filtre gaussien pour démo
    deblurred = cv2.GaussianBlur(image, (5, 5), 0)
    return deblurred

def display_score_box(score, label):
    """Affiche un score dans une boîte colorée"""
    if score < 20:
        st.markdown(f"""
        <div class="success-box">
            <h3>✅ {label}: {score}/100</h3>
            <p>Niveau: Très faible (Probablement authentique)</p>
        </div>
        """, unsafe_allow_html=True)
    elif score < 40:
        st.markdown(f"""
        <div class="success-box">
            <h3>ℹ️ {label}: {score}/100</h3>
            <p>Niveau: Faible (Probablement authentique)</p>
        </div>
        """, unsafe_allow_html=True)
    elif score < 60:
        st.markdown(f"""
        <div class="warning-box">
            <h3>⚠️ {label}: {score}/100</h3>
            <p>Niveau: Moyen (Incertain)</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="danger-box">
            <h3>🚨 {label}: {score}/100</h3>
            <p>Niveau: Élevé (Probablement manipulée)</p>
        </div>
        """, unsafe_allow_html=True)

# ========== HEADER ==========
st.markdown('<p class="main-header">🔍 DeblurGAN + Forensic Analysis</p>', unsafe_allow_html=True)

st.markdown("""
Cette application combine **DeblurGAN** pour le défloutage d'images avec une **analyse forensique** 
pour détecter les artefacts de manipulation.

**Fonctionnalités :**
- 🎯 Défloutage d'images avec GAN
- 🔬 Analyse forensique (ELA + Autoencoder)
- 📊 Score de manipulation (0-100)
""")

st.markdown("---")

# ========== SIDEBAR ==========
st.sidebar.title("⚙️ Configuration")
mode = st.sidebar.radio(
    "Mode d'utilisation",
    ["Défloutage Simple", "Analyse Forensique Seule", "Pipeline Complet"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Astuce :** Uploadez une image pour commencer!")

# ========== MODE 1: DÉFLOUTAGE SIMPLE ==========
if mode == "Défloutage Simple":
    st.header("📸 Défloutage d'Image")
    
    uploaded_file = st.file_uploader(
        "Choisissez une image floue",
        type=['png', 'jpg', 'jpeg'],
        key="deblur_upload"
    )
    
    if uploaded_file is not None:
        # Lecture de l'image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Conversion RGB -> BGR pour OpenCV
        if len(image_np.shape) == 3:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_np
        
        # Affichage
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Image Originale (Floue)")
            st.image(image, use_container_width=True)
            st.caption(f"Dimensions: {image_np.shape[1]}x{image_np.shape[0]}")
        
        # Bouton de défloutage
        if st.button("🚀 Déflouter l'image", type="primary", use_container_width=True):
            with st.spinner("Défloutage en cours..."):
                # Défloutage
                deblurred_bgr = deblur_image(image_bgr)
                deblurred_rgb = cv2.cvtColor(deblurred_bgr, cv2.COLOR_BGR2RGB)
                
                # Affichage
                with col2:
                    st.subheader("Image Défloutée")
                    st.image(deblurred_rgb, use_container_width=True)
                    st.caption("Résultat du défloutage")
                
                st.success("✅ Défloutage terminé avec succès!")
                
                # Bouton de téléchargement
                st.download_button(
                    label="📥 Télécharger l'image défloutée",
                    data=cv2.imencode('.png', deblurred_bgr)[1].tobytes(),
                    file_name="deblurred_image.png",
                    mime="image/png"
                )

# ========== MODE 2: ANALYSE FORENSIQUE ==========
elif mode == "Analyse Forensique Seule":
    st.header("🔬 Analyse Forensique")
    
    uploaded_file = st.file_uploader(
        "Choisissez une image à analyser",
        type=['png', 'jpg', 'jpeg'],
        key="forensic_upload"
    )
    
    if uploaded_file is not None:
        # Lecture
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Affichage
        st.subheader("Image à Analyser")
        st.image(image, use_container_width=True)
        
        if st.button("🔍 Analyser", type="primary", use_container_width=True):
            with st.spinner("Analyse en cours..."):
                # Analyse forensique
                results = detector.analyze_image(image_bgr, include_details=True)
                
                st.markdown("---")
                
                # Score principal
                display_score_box(results['tampering_score'], "Score de Manipulation")
                
                # Métriques détaillées
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Score ELA", f"{results['ela_score']:.3f}")
                
                with col2:
                    if results.get('autoencoder_score'):
                        st.metric("Score Autoencoder", f"{results['autoencoder_score']:.3f}")
                    else:
                        st.metric("Score Autoencoder", "N/A")
                
                with col3:
                    st.metric("Manipulée ?", "Oui" if results['is_tampered'] else "Non")
                
                # Verdict
                st.markdown("---")
                st.subheader("🎯 Verdict")
                if results['is_tampered']:
                    st.error("🚨 **ALERTE :** Cette image présente des signes de manipulation!")
                else:
                    st.success("✅ **AUTHENTIQUE :** Aucun signe significatif de manipulation détecté.")

# ========== MODE 3: PIPELINE COMPLET ==========
elif mode == "Pipeline Complet":
    st.header("🔗 Pipeline Complet : Défloutage + Forensic")
    
    uploaded_file = st.file_uploader(
        "Choisissez une image floue",
        type=['png', 'jpg', 'jpeg'],
        key="complete_upload"
    )
    
    if uploaded_file is not None:
        # Lecture
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Affichage image originale
        st.subheader("📷 Image Originale (Floue)")
        st.image(image, use_container_width=True)
        
        if st.button("🚀 Lancer le Pipeline Complet", type="primary", use_container_width=True):
            
            # ========== ÉTAPE 1: DÉFLOUTAGE ==========
            st.markdown("---")
            st.subheader("🎨 Étape 1 : Défloutage")
            
            with st.spinner("Défloutage en cours..."):
                deblurred_bgr = deblur_image(image_bgr)
                deblurred_rgb = cv2.cvtColor(deblurred_bgr, cv2.COLOR_BGR2RGB)
            
            st.success("✅ Défloutage terminé")
            
            # Affichage comparatif
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Avant", use_container_width=True)
            with col2:
                st.image(deblurred_rgb, caption="Après", use_container_width=True)
            
            # ========== ÉTAPE 2: ANALYSE FORENSIQUE ==========
            st.markdown("---")
            st.subheader("🔬 Étape 2 : Analyse Forensique")
            
            with st.spinner("Analyse forensique en cours..."):
                # Analyse des deux images
                results_original = detector.analyze_image(image_bgr)
                results_deblurred = detector.analyze_image(deblurred_bgr)
            
            st.success("✅ Analyse terminée")
            
            # Comparaison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Image Originale")
                st.metric("Score", f"{results_original['tampering_score']}/100")
                st.caption(results_original['confidence_level'])
            
            with col2:
                st.markdown("#### Image Défloutée")
                st.metric("Score", f"{results_deblurred['tampering_score']}/100")
                st.caption(results_deblurred['confidence_level'])
            
            # Delta
            score_diff = results_deblurred['tampering_score'] - results_original['tampering_score']
            
            st.markdown("---")
            st.subheader("📊 Analyse Comparative")
            
            if score_diff > 30:
                st.error(f"🚨 **ARTEFACTS MAJEURS** : Le défloutage a introduit d'importants artefacts (+{score_diff} points)")
            elif score_diff > 15:
                st.warning(f"⚠️ **ARTEFACTS MODÉRÉS** : Quelques artefacts détectés (+{score_diff} points)")
            elif results_deblurred['tampering_score'] < 30:
                st.success(f"✅ **SUCCÈS** : Défloutage réussi avec peu d'artefacts (score: {results_deblurred['tampering_score']}/100)")
            else:
                st.info(f"ℹ️ **INCERTAIN** : Score de {results_deblurred['tampering_score']}/100, révision manuelle recommandée")
            
            # Bouton téléchargement
            st.markdown("---")
            st.download_button(
                label="📥 Télécharger l'image défloutée",
                data=cv2.imencode('.png', deblurred_bgr)[1].tobytes(),
                file_name="deblurred_forensic.png",
                mime="image/png"
            )

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>DeblurGAN Forensic Analysis</strong> | Projet INPT 2025</p>
    <p>Powered by DeblurGAN + FastAPI + Streamlit + MLflow</p>
</div>
""", unsafe_allow_html=True)