import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="Analyse d'Intensité - Prétraitement d'Images",
    page_icon="🎨",
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
</style>
""", unsafe_allow_html=True)

# ============================================================
# FONCTIONS DE PRÉTRAITEMENT
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
        # Pour RGB: convertir en YUV, égaliser Y, reconvertir
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    else:
        return cv2.equalizeHist(img)

def equalize_clahe(img, clip_limit=2.0, grid_size=(8, 8)):
    """Égalisation adaptative CLAHE"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    if len(img.shape) == 3:
        # Pour RGB: convertir en YUV, appliquer CLAHE sur Y, reconvertir
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    else:
        return clahe.apply(img)

def apply_preprocessing(img, method_name):
    """Applique la méthode de prétraitement choisie"""
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
# FONCTIONS POUR LES HISTOGRAMMES
# ============================================================

def plot_histogram(img, title, ax):
    """Affiche l'histogramme d'une image"""
    if len(img.shape) == 3:
        # Image RGB - afficher les 3 canaux
        colors = ('r', 'g', 'b')
        labels = ('Rouge', 'Vert', 'Bleu')
        for i, (color, label) in enumerate(zip(colors, labels)):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            ax.plot(hist, color=color, alpha=0.7, label=label, linewidth=1.5)
    else:
        # Image niveaux de gris
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

def display_image_with_histogram(img, method_name, col):
    """Affiche une image et son histogramme dans une colonne"""
    with col:
        st.markdown(f"**{method_name}**")
        
        # Afficher l'image
        if len(img.shape) == 3:
            st.image(img, use_container_width=True, channels="RGB")
        else:
            st.image(img, use_container_width=True, clamp=True)
        
        # Créer l'histogramme
        fig, ax = plt.subplots(figsize=(4, 3))
        plot_histogram(img, "", ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ============================================================
# FONCTION DE TRAITEMENT PRINCIPALE
# ============================================================

def process_single_image(img, image_name):
    """Traite une seule image avec les 4 stratégies et affiche les résultats"""
    st.markdown(f"### 🖼️ {image_name}")
    
    # Appliquer les 4 stratégies
    methods = ["Originale (Brute)", "Normalisation Min-Max", 
               "Égalisation Globale (GHE)", "Égalisation Adaptative (CLAHE)"]
    
    processed_images = {}
    for method in methods:
        processed_images[method] = apply_preprocessing(img, method)
    
    # Afficher dans 4 colonnes
    cols = st.columns(4)
    for col, method in zip(cols, methods):
        display_image_with_histogram(processed_images[method], method, col)
    
    # Ajouter une ligne de séparation
    st.markdown("---")
    
    return processed_images

# ============================================================
# FONCTION POUR AFFICHER LE RÉCAPITULATIF
# ============================================================

def display_summary(all_processed_images):
    """Affiche un tableau récapitulatif des informations sur les images"""
    st.markdown("## 📊 Récapitulatif des images traitées")
    
    summary_data = []
    for img_name, methods_dict in all_processed_images.items():
        for method_name, img in methods_dict.items():
            if len(img.shape) == 3:
                mode = "RGB"
                mean_val = np.mean(img, axis=(0, 1))
                mean_str = f"R:{mean_val[0]:.1f}, V:{mean_val[1]:.1f}, B:{mean_val[2]:.1f}"
                std_val = np.std(img, axis=(0, 1))
                std_str = f"R:{std_val[0]:.1f}, V:{std_val[1]:.1f}, B:{std_val[2]:.1f}"
            else:
                mode = "Gris"
                mean_val = np.mean(img)
                mean_str = f"{mean_val:.1f}"
                std_val = np.std(img)
                std_str = f"{std_val:.1f}"
            
            summary_data.append({
                "Image": img_name,
                "Méthode": method_name,
                "Mode": mode,
                "Moyenne": mean_str,
                "Écart-type": std_str,
                "Min": np.min(img),
                "Max": np.max(img)
            })
    
    df = pd.DataFrame(summary_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

# ============================================================
# APPLICATION PRINCIPALE
# ============================================================

def main():
    # En-tête
    st.markdown('<div class="main-header"><h1 style="color:white; text-align:center;">🎨 Analyse d\'Intensité</h1><h3 style="color:white; text-align:center;">Comparaison des stratégies de prétraitement d\'images</h3></div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Paramètres")
        
        st.subheader("📤 Chargement des images")
        uploaded_files = st.file_uploader(
            "Chargez vos images",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Vous pouvez charger plusieurs images à la fois"
        )
        
        st.markdown("---")
        st.subheader("🔬 Méthodes appliquées")
        st.markdown("""
        - ✅ **Originale (Brute)** : Image sans modification
        - ✅ **Normalisation Min-Max** : Redimensionnement des intensités entre 0 et 255
        - ✅ **Égalisation Globale (GHE)** : Distribution uniforme des intensités
        - ✅ **Égalisation Adaptative (CLAHE)** : Égalisation locale avec limitation de contraste
        """)
        
        st.markdown("---")
        st.subheader("ℹ️ Information")
        st.info("""
        **Comment ça marche ?**
        1. Chargez vos images
        2. Chaque image est traitée avec les 4 méthodes
        3. Visualisez les résultats et histogrammes
        4. Comparez l'impact sur la distribution des intensités
        """)
    
    # Corps principal
    if not uploaded_files:
        st.info("👈 **Commencez par charger des images dans la barre latérale gauche**")
        
        # Afficher un exemple
        col1, col2, col3 = st.columns(3)
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background: #f0f2f6; border-radius: 10px;">
                <h3>📸 Exemple</h3>
                <p>Chargez vos propres images pour voir<br>l'effet des différentes méthodes de prétraitement</p>
                <p style="color: #667eea;">⬅️ Cliquez sur "upload" pour commencer</p>
            </div>
            """, unsafe_allow_html=True)
        return
    
    # Nombre d'images chargées
    st.success(f"✅ {len(uploaded_files)} image(s) chargée(s) avec succès !")
    
    # Chargement et traitement des images
    all_processed_images = {}
    
    for idx, file in enumerate(uploaded_files):
        # Charger l'image
        pil_img = Image.open(file)
        img_array = np.array(pil_img)
        
        # Convertir en RGB si nécessaire
        if len(img_array.shape) == 2:
            # Image en niveaux de gris
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            st.info(f"📷 {file.name} : Image en niveaux de gris convertie en RGB pour l'affichage")
        elif img_array.shape[2] == 4:
            # Image RGBA
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            st.info(f"📷 {file.name} : Image RGBA convertie en RGB")
        else:
            img_rgb = img_array
        
        # Redimensionner pour un affichage standard (optionnel)
        h, w = img_rgb.shape[:2]
        if max(h, w) > 600:
            scale = 600 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_rgb = cv2.resize(img_rgb, (new_w, new_h))
        
        # Traiter l'image
        processed = process_single_image(img_rgb, file.name)
        all_processed_images[file.name] = processed
    
    # Afficher le récapitulatif
    st.markdown("---")
    display_summary(all_processed_images)
    
    # Graphique comparatif des histogrammes
    st.markdown("## 📈 Comparaison globale des histogrammes")
    
    # Sélection de l'image pour la comparaison
    if len(uploaded_files) > 0:
        selected_image = st.selectbox(
            "Choisissez une image pour comparer les histogrammes des 4 méthodes :",
            list(all_processed_images.keys())
        )
        
        methods = ["Originale (Brute)", "Normalisation Min-Max", 
                   "Égalisation Globale (GHE)", "Égalisation Adaptative (CLAHE)"]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, method in enumerate(methods):
            img = all_processed_images[selected_image][method]
            
            if len(img.shape) == 3:
                # Pour RGB, afficher le canal de luminance
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                axes[idx].plot(hist, color='black', linewidth=2)
                axes[idx].fill_between(range(256), hist[:, 0], alpha=0.3, color='blue')
            else:
                hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                axes[idx].plot(hist, color='black', linewidth=2)
                axes[idx].fill_between(range(256), hist[:, 0], alpha=0.3, color='blue')
            
            axes[idx].set_xlabel('Intensité des pixels', fontsize=10)
            axes[idx].set_ylabel('Nombre de pixels', fontsize=10)
            axes[idx].set_title(method, fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_xlim(0, 255)
        
        plt.suptitle(f'Comparaison des histogrammes - {selected_image}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    
    # Pied de page
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Analyse d'Intensité - Vision par Ordinateur | DASSI M. Léa | MELONG Lethycia | NGUEFACK T. Arthur</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
