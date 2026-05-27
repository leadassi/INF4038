# 🎨 TP Vision par Ordinateur - Analyse d'Intensité et Prétraitement d'Images

---

## 📌 Informations générales

| | |
|---|---|
| **Module** | Vision par Ordinateur |
| **Proposé par** | DASSI M. Léa, MELONG Lethycia, NGUEFACK T. Arthur |
| **Date** | 25 mai 2026 |
| **Langage** | Python 3.8+ |
| **Framework** | Streamlit |

---

## 🎯 Présentation du TP

Ce TP pratique a pour objectif d'illustrer l'impact des méthodes de prétraitement d'intensité sur la robustesse des descripteurs globaux en vision par ordinateur.

L'application permet de :
- Modifier l'éclairage d'une image (luminosité, contraste, gamma)
- Appliquer différentes méthodes d'égalisation d'histogramme
- Extraire des descripteurs globaux (HOG, LBP, histogrammes de couleurs)
- Comparer la robustesse des descripteurs face aux variations d'éclairage

---

## 📚 Objectifs pédagogiques

| Objectif | Description |
|----------|-------------|
| 🎯 **Comprendre** | L'impact des variations d'illumination sur les images numériques |
| 🔧 **Implémenter** | Les principales méthodes de normalisation et d'égalisation |
| 📊 **Analyser** | Les histogrammes et leur distribution |
| 🔬 **Comparer** | Les performances de GHE vs CLAHE vs Min-Max |
| 📈 **Évaluer** | La robustesse des descripteurs globaux |

---

## ✨ Fonctionnalités

### 1. Modification d'éclairage

| Type | Plage | Description |
|------|-------|-------------|
| 🌞 **Luminosité** | -150 à 150 | Assombrir ou éclaircir l'image |
| 🎨 **Contraste** | -40 à 40 | Augmenter ou diminuer le contraste |
| 📷 **Gamma** | -80 à 80 | Correction d'exposition non linéaire |

### 2. Méthodes d'égalisation

| Méthode | Description |
|---------|-------------|
| **Originale (Brute)** | Image sans modification (référence) |
| **Normalisation Min-Max** | Redimensionnement linéaire des intensités |
| **Égalisation Globale (GHE)** | Distribution uniforme sur toute l'image |
| **Égalisation Adaptative (CLAHE)** | Égalisation locale avec limitation de contraste |

### 3. Descripteurs globaux

| Descripteur | Dimension | Description |
|-------------|-----------|-------------|
| **Histogramme de couleurs** | 96 (32×3) | Distribution des couleurs RGB |
| **LBP** (Local Binary Patterns) | 10 (uniform) | Descripteur de texture |
| **HOG** (Histogram of Oriented Gradients) | Variable | Descripteur de forme |

### 4. Visualisations

- 📊 Affichage côte à côte des images (originale, éclairée, égalisées)
- 📈 Histogrammes des intensités pour chaque image
- 📉 Graphiques de comparaison des descripteurs
- 📋 Tableaux récapitulatifs des similarités

---

## 🛠️ Installation

### Étape 1 : Installer Python

Téléchargez et installez Python depuis [python.org](https://www.python.org/downloads/)

Vérifiez l'installation :
```bash
python --version
```
### Étape 2 : Télécharger le code du TP

```bash
# Cloner le dépôt (ou télécharger le fichier ZIP)
git clone https://github.com/leadassi/INF4038.git
cd INF4038
```

### Étape 3 : Créer un environnement virtuel

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### Étape 4 : Installer les dépendances

```bash
pip install -r requirements.txt
```

### Étape 5 : Lancer l'application

```bash
streamlit run app.py
```
L'application s'ouvre automatiquement dans votre navigateur à l'adresse : http://localhost:8501

