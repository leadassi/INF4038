import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from skimage.feature import local_binary_pattern, hog
from skimage import exposure

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="Analyse d'Intensité - TP Vision par Ordinateur",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .sub-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .similarity-high {
        background-color: #d4edda;
        color: #155724;
        padding: 0.2rem 0.5rem;
        border-radius: 20px;
        font-weight: bold;
    }
    .similarity-low {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.2rem 0.5rem;
        border-radius: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# FONCTIONS DE MODIFICATION D'ÉCLAIRAGE
# ============================================================

def adjust_brightness(img, value):
    """Ajuste la luminosité de l'image"""
    if len(img.shape) == 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    else:
        return np.clip(img.astype(np.float32) + value, 0, 255).astype(np.uint8)

def adjust_contrast(img, factor):
    """Ajuste le contraste de l'image"""
    mean = np.mean(img)
    if len(img.shape) == 3:
        adjusted = mean + factor * (img - mean)
    else:
        adjusted = mean + factor * (img - mean)
    return np.clip(adjusted, 0, 255).astype(np.uint8)

def adjust_gamma(img, gamma):
    """Correction gamma pour simuler différents éclairages"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)

def apply_lighting(img, lighting_type, intensity):
    """Applique une modification d'éclairage selon le type choisi"""
    if lighting_type == "Luminosité":
        return adjust_brightness(img, intensity)
    elif lighting_type == "Contraste":
        factor = 0.5 + (intensity / 50)
        return adjust_contrast(img, factor)
    elif lighting_type == "Gamma (exposition)":
        gamma = 0.5 + (intensity / 100)
        return adjust_gamma(img, gamma)
    else:
        return img

# ============================================================
# FONCTIONS DE PRÉTRAITEMENT (ÉGALISATION)
# ============================================================

def normalize_minmax(img):
    """Normalisation Min-Max simple"""
    img = img.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 0:
        img = (img - img_min) / (img_max - img_min) * 255
    return np.clip(img, 0, 255).astype(np.uint8)

def equalize_histogram_ghe(img):
    """Égalisation d'histogramme globale (GHE)"""
    if len(img.shape) == 3:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    else:
        return cv2.equalizeHist(img)

def equalize_clahe(img, clip_limit=2.0, grid_size=(8, 8)):
    """Égalisation adaptative CLAHE"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    if len(img.shape) == 3:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    else:
        return clahe.apply(img)

def apply_equalization(img, method_name):
    """Applique la méthode d'égalisation choisie"""
    if method_name == "Originale (Brute)":
        return img
    elif method_name == "Normalisation Min-Max":
        return normalize_minmax(img)
    elif method_name == "Égalisation Globale (GHE)":
        return equalize_histogram_ghe(img)
    elif method_name == "Égalisation Adaptative (CLAHE)":
        return equalize_clahe(img)
    else:
        return img

# ============================================================
# FONCTIONS D'EXTRACTION DE DESCRIPTEURS GLOBAUX
# ============================================================

def extract_color_histogram(img, bins=32):
    """Extrait l'histogramme de couleurs (RGB)"""
    if len(img.shape) == 3:
        hist_r = cv2.calcHist([img], [0], None, [bins], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [bins], [0, 256])
        hist_b = cv2.calcHist([img], [2], None, [bins], [0, 256])
        
        # Normalisation
        hist_r = hist_r / np.sum(hist_r)
        hist_g = hist_g / np.sum(hist_g)
        hist_b = hist_b / np.sum(hist_b)
        
        return np.concatenate([hist_r.flatten(), hist_g.flatten(), hist_b.flatten()])
    else:
        hist = cv2.calcHist([img], [0], None, [bins], [0, 256])
        hist = hist / np.sum(hist)
        return hist.flatten()

def extract_lbp(img, P=8, R=1, method='uniform'):
    """Extrait le descripteur LBP (Local Binary Pattern)"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    lbp = local_binary_pattern(gray, P, R, method)
    n_bins = P + 2 if method == 'uniform' else 256
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist / np.sum(hist)
    return hist

def extract_hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """Extrait le descripteur HOG (Histogram of Oriented Gradients)"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    hog_features = hog(gray, orientations=orientations, 
                       pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block, 
                       visualize=False)
    return hog_features

def extract_all_descriptors(img, selected_descriptors):
    """Extrait les descripteurs sélectionnés"""
    descriptors = {}
    
    if "Histogramme de couleurs" in selected_descriptors:
        descriptors["Histogramme de couleurs"] = extract_color_histogram(img)
    
    if "LBP" in selected_descriptors:
        descriptors["LBP"] = extract_lbp(img)
    
    if "HOG" in selected_descriptors:
        descriptors["HOG"] = extract_hog(img)
    
    return descriptors

# ============================================================
# FONCTIONS DE COMPARAISON
# ============================================================

def compare_descriptors(desc1, desc2, descriptor_name):
    """Compare deux descripteurs et retourne la similarité"""
    if len(desc1) != len(desc2):
        return 0.0
    
    # Distance cosinus
    dot_product = np.dot(desc1, desc2)
    norm1 = np.linalg.norm(desc1)
    norm2 = np.linalg.norm(desc2)
    
    if norm1 > 0 and norm2 > 0:
        similarity = dot_product / (norm1 * norm2)
    else:
        similarity = 0.0
    
    return similarity * 100  # Retourne un pourcentage

def get_similarity_class(similarity):
    """Retourne la classe CSS en fonction de la similarité"""
    if similarity >= 80:
        return "similarity-high"
    elif similarity >= 60:
        return ""
    else:
        return "similarity-low"

# ============================================================
# AFFICHAGE DES HISTOGRAMMES
# ============================================================

def plot_histogram(img, title, ax):
    """Affiche l'histogramme d'une image"""
    if len(img.shape) == 3:
        colors = ('r', 'g', 'b')
        labels = ('Rouge', 'Vert', 'Bleu')
        for i, (color, label) in enumerate(zip(colors, labels)):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            ax.plot(hist, color=color, alpha=0.7, label=label, linewidth=1.5)
    else:
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        ax.plot(hist, color='black', alpha=0.8, linewidth=1.5)
        ax.fill_between(range(256), hist[:, 0], alpha=0.3, color='gray')
    
    ax.set_xlabel('Intensité des pixels', fontsize=10)
    ax.set_ylabel('Nombre de pixels', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if len(img.shape) == 3:
        ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, 255)

# ============================================================
# APPLICATION PRINCIPALE
# ============================================================

def main():
    # En-tête
    st.markdown('<div class="main-header"><h1 style="color:white; text-align:center;">TP Vision par Ordinateur</h1><h3 style="color:white; text-align:center;">Analyse d\'Intensité - Égalisation et Extraction de Descripteurs Globaux</h3></div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Paramètres")
        
        st.subheader("📤 Chargement de l'image")
        uploaded_file = st.file_uploader(
            "Chargez une image",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Sélectionnez une image pour l'analyse"
        )
        
        st.markdown("---")
        
        st.subheader("💡 Modification d'éclairage")
        lighting_type = st.selectbox(
            "Type de modification",
            ["Luminosité", "Contraste", "Gamma (exposition)"],
            help="Choisissez le type d'altération de l'éclairage"
        )
        
        if lighting_type == "Luminosité":
            intensity = st.slider("Intensité", -100, 100, 50, help="Négatif = plus sombre, Positif = plus clair")
        elif lighting_type == "Contraste":
            intensity = st.slider("Intensité", -40, 40, 20, help="Négatif = contraste faible, Positif = contraste fort")
        else:
            intensity = st.slider("Gamma", 10, 200, 150, help="<100 = plus clair, >100 = plus sombre")
            intensity = intensity - 100
        
        st.markdown("---")
        
        st.subheader("🔬 Méthode d'égalisation")
        equalization_method = st.selectbox(
            "Choisissez la méthode à appliquer sur les deux images",
            ["Originale (Brute)", "Normalisation Min-Max", "Égalisation Globale (GHE)", "Égalisation Adaptative (CLAHE)"]
        )
        
        st.markdown("---")
        
        st.subheader("📊 Descripteurs globaux à extraire")
        selected_descriptors = st.multiselect(
            "Choisissez les descripteurs",
            ["Histogramme de couleurs", "LBP", "HOG"],
            default=["Histogramme de couleurs", "LBP", "HOG"]
        )
        
        st.markdown("---")
        st.info("""
        **🎯 Objectif du TP :**
        1. Modifier l'éclairage d'une image
        2. Appliquer la MÊME méthode d'égalisation sur les DEUX images
        3. Extraire les descripteurs globaux des DEUX images égalisées
        4. Comparer la robustesse des descripteurs
        """)
    
    # Corps principal
    if uploaded_file is None:
        st.info("👈 **Chargez une image dans la barre latérale pour commencer**")
        
        # Afficher un exemple
        col1, col2, col3 = st.columns(3)
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background: #f0f2f6; border-radius: 10px;">
                <h3>📸 Instructions</h3>
                <p>1. Chargez une image<br>
                2. Modifiez l'éclairage<br>
                3. Choisissez une méthode d'égalisation<br>
                4. La méthode est appliquée aux DEUX images<br>
                5. Comparez les descripteurs extraits</p>
            </div>
            """, unsafe_allow_html=True)
        return
    
    # Chargement de l'image
    pil_img = Image.open(uploaded_file)
    img_original = np.array(pil_img)
    
    # Conversion en RGB si nécessaire
    if len(img_original.shape) == 2:
        img_original = cv2.cvtColor(img_original, cv2.COLOR_GRAY2RGB)
        st.info("📷 Image en niveaux de gris convertie en RGB")
    elif img_original.shape[2] == 4:
        img_original = cv2.cvtColor(img_original, cv2.COLOR_RGBA2RGB)
        st.info("📷 Image RGBA convertie en RGB")
    
    # Redimensionnement pour l'affichage
    h, w = img_original.shape[:2]
    if max(h, w) > 500:
        scale = 500 / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_original = cv2.resize(img_original, (new_w, new_h))
    
    # ÉTAPE 1: Application de la modification d'éclairage
    img_lighting = apply_lighting(img_original, lighting_type, intensity)
    
    # ÉTAPE 2: Application de la MÊME méthode d'égalisation sur les DEUX images
    img_original_equalized = apply_equalization(img_original, equalization_method)
    img_lighting_equalized = apply_equalization(img_lighting, equalization_method)
    
    # ÉTAPE 3: Extraction des descripteurs sur les DEUX images égalisées
    descriptors_original = extract_all_descriptors(img_original_equalized, selected_descriptors)
    descriptors_lighting = extract_all_descriptors(img_lighting_equalized, selected_descriptors)
    
    # ==================== AFFICHAGE DES IMAGES ====================
    st.markdown("## 🖼️ Pipeline de traitement")
    
    # Ligne 1: Images originales (avant égalisation)
    st.markdown("### Étape 1: Images avant égalisation")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="result-card"><p style="text-align:center"><b>📸 Image originale</b></p></div>', unsafe_allow_html=True)
        st.image(img_original, use_container_width=True, channels="RGB")
        
        fig, ax = plt.subplots(figsize=(5, 3))
        plot_histogram(img_original, "Histogramme", ax)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown(f'<div class="result-card"><p style="text-align:center"><b>💡 Image après modification d\'éclairage</b><br>({lighting_type}: {intensity})</p></div>', unsafe_allow_html=True)
        st.image(img_lighting, use_container_width=True, channels="RGB")
        
        fig, ax = plt.subplots(figsize=(5, 3))
        plot_histogram(img_lighting, "Histogramme", ax)
        st.pyplot(fig)
        plt.close()
    
    # Ligne 2: Images après égalisation
    st.markdown(f"### Étape 2: Images après application de **{equalization_method}**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="result-card"><p style="text-align:center"><b>🔬 Image originale égalisée</b></p></div>', unsafe_allow_html=True)
        st.image(img_original_equalized, use_container_width=True, channels="RGB")
        
        fig, ax = plt.subplots(figsize=(5, 3))
        plot_histogram(img_original_equalized, "Histogramme après égalisation", ax)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown(f'<div class="result-card"><p style="text-align:center"><b>🔬 Image éclairée égalisée</b></p></div>', unsafe_allow_html=True)
        st.image(img_lighting_equalized, use_container_width=True, channels="RGB")
        
        fig, ax = plt.subplots(figsize=(5, 3))
        plot_histogram(img_lighting_equalized, "Histogramme après égalisation", ax)
        st.pyplot(fig)
        plt.close()
    
    # ==================== EXTRACTION DES DESCRIPTEURS ====================
    st.markdown("---")
    st.markdown("## 📊 Extraction des descripteurs globaux")
    
    if not selected_descriptors:
        st.warning("⚠️ Veuillez sélectionner au moins un descripteur dans la barre latérale")
    else:
        st.markdown(f"*Méthode d'égalisation appliquée :* **{equalization_method}**")
        
        # Afficher les dimensions des descripteurs
        st.markdown("### Dimensions des descripteurs extraits")
        
        dim_data = []
        for desc_name in selected_descriptors:
            dim_data.append({
                "Descripteur": desc_name,
                "Dimension": len(descriptors_original[desc_name])
            })
        
        st.dataframe(pd.DataFrame(dim_data), use_container_width=True, hide_index=True)
        
        # Visualisation détaillée des descripteurs
        st.markdown("### Visualisation des descripteurs")
        
        for desc_name in selected_descriptors:
            with st.expander(f"📈 {desc_name} - Détails"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔬 Image originale égalisée**")
                    desc = descriptors_original[desc_name]
                    st.write(f"Dimension: {len(desc)}")
                    st.write(f"Premières 10 valeurs: {desc[:10].round(4)}")
                    st.write(f"Statistiques: min={desc.min():.4f}, max={desc.max():.4f}, mean={desc.mean():.4f}, std={desc.std():.4f}")
                    
                    # Graphique pour histogramme de couleurs
                    if desc_name == "Histogramme de couleurs":
                        fig, ax = plt.subplots(figsize=(10, 3))
                        ax.plot(desc, color='blue', alpha=0.7)
                        ax.set_title(f"{desc_name} - Image originale égalisée")
                        ax.set_xlabel("Bins")
                        ax.set_ylabel("Fréquence")
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close()
                
                with col2:
                    st.markdown("**🔬 Image éclairée égalisée**")
                    desc = descriptors_lighting[desc_name]
                    st.write(f"Dimension: {len(desc)}")
                    st.write(f"Premières 10 valeurs: {desc[:10].round(4)}")
                    st.write(f"Statistiques: min={desc.min():.4f}, max={desc.max():.4f}, mean={desc.mean():.4f}, std={desc.std():.4f}")
                    
                    # Graphique pour histogramme de couleurs
                    if desc_name == "Histogramme de couleurs":
                        fig, ax = plt.subplots(figsize=(10, 3))
                        ax.plot(desc, color='orange', alpha=0.7)
                        ax.set_title(f"{desc_name} - Image éclairée égalisée")
                        ax.set_xlabel("Bins")
                        ax.set_ylabel("Fréquence")
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close()
        
        # ==================== COMPARAISON DES DESCRIPTEURS ====================
        st.markdown("---")
        st.markdown("## 📈 Comparaison des descripteurs")
        st.markdown(f"*Comparaison entre l'image originale égalisée et l'image éclairée égalisée (méthode: {equalization_method})*")
        
        # Tableau de comparaison
        comparison_data = []
        for desc_name in selected_descriptors:
            similarity = compare_descriptors(
                descriptors_original[desc_name], 
                descriptors_lighting[desc_name],
                desc_name
            )
            
            # Interprétation
            if similarity >= 85:
                interpretation = "✅ Très robuste - Variations d'éclairage bien compensées"
            elif similarity >= 70:
                interpretation = "👍 Robuste - Bonne compensation des variations"
            elif similarity >= 50:
                interpretation = "⚠️ Moyennement robuste - Compensation partielle"
            else:
                interpretation = "❌ Peu robuste - Méthode inefficace pour ce type de variation"
            
            comparison_data.append({
                "Descripteur": desc_name,
                "Similarité (%)": f"{similarity:.1f}%",
                "Interprétation": interpretation
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)
        
        # Graphique de comparaison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        similarities = [float(compare_descriptors(descriptors_original[d], descriptors_lighting[d], d)) for d in selected_descriptors]
        colors = ['#4ECDC4' if s >= 70 else '#FF6B6B' if s < 50 else '#FFB347' for s in similarities]
        
        bars = ax.bar(selected_descriptors, similarities, color=colors, edgecolor='black', linewidth=1.5)
        
        # Ajouter les valeurs sur les barres
        for bar, sim in zip(bars, similarities):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{sim:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax.set_ylabel('Similarité (%)', fontsize=12)
        ax.set_xlabel('Descripteurs', fontsize=12)
        ax.set_title(f'Robustesse des descripteurs face aux variations d\'éclairage\nMéthode d\'égalisation : {equalization_method}', fontsize=14)
        ax.set_ylim(0, 105)
        ax.axhline(y=70, color='green', linestyle='--', alpha=0.7, label='Seuil de robustesse (70%)')
        ax.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Seuil de tolérance (50%)')
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        
        st.pyplot(fig)
        plt.close()
        
        # ==================== CONCLUSION ====================
        st.markdown("---")
        st.markdown("## 📝 Analyse et conclusion")
        
        # Calcul du score moyen
        avg_similarity = np.mean(similarities)
        
        if avg_similarity >= 80:
            conclusion_level = "🌟 EXCELLENTE"
            conclusion_color = "#28a745"
        elif avg_similarity >= 65:
            conclusion_level = "👍 BONNE"
            conclusion_color = "#17a2b8"
        elif avg_similarity >= 50:
            conclusion_level = "⚠️ MOYENNE"
            conclusion_color = "#ffc107"
        else:
            conclusion_level = "❌ FAIBLE"
            conclusion_color = "#dc3545"
        
        st.markdown(f"""
        <div class="result-card">
        <h4>🔍 Récapitulatif de l'expérience :</h4>
        <ul>
            <li><b>Modification d'éclairage appliquée</b> : {lighting_type} (intensité: {intensity})</li>
            <li><b>Méthode d'égalisation</b> : {equalization_method}</li>
            <li><b>Descripteurs extraits</b> : {', '.join(selected_descriptors)}</li>
            <li><b>Similarité moyenne</b> : {avg_similarity:.1f}%</li>
            <li><b>Robustesse globale</b> : <span style="color:{conclusion_color}; font-weight:bold;">{conclusion_level}</span></li>
        </ul>
        
        <h4>💡 Interprétation :</h4>
        <ul>
            <li>La similarité mesure à quel point les descripteurs sont invariants aux modifications d'éclairage</li>
            <li>Une similarité élevée (>70%) indique que la méthode d'égalisation est efficace pour ce type de variation</li>
            <li>CLAHE est généralement plus robuste que GHE car il préserve les détails locaux</li>
            <li>HOG est souvent plus robuste que les histogrammes de couleurs face aux variations de luminosité</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Bonus : Export des résultats
        with st.expander("📤 Exporter les résultats (JSON)"):
            import json
            
            export_data = {
                "image_name": uploaded_file.name,
                "lighting_modification": {
                    "type": lighting_type,
                    "intensity": intensity
                },
                "equalization_method": equalization_method,
                "descriptors_extracted": selected_descriptors,
                "similarity_scores": {d: float(compare_descriptors(descriptors_original[d], descriptors_lighting[d], d)) for d in selected_descriptors},
                "average_similarity": float(avg_similarity)
            }
            
            st.json(export_data)
        
        # Téléchargement
        json_str = json.dumps(export_data, indent=2)
        st.download_button(
            label="📥 Télécharger les résultats (JSON)",
            data=json_str,
            file_name=f"resultats_{uploaded_file.name.split('.')[0]}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()
