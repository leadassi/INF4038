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

## 🎯 Objectifs du TP

À l'issue de ce TP, vous serez capable de :

1. **Comprendre** l'impact des variations d'illumination sur les images numériques
2. **Implémenter** les principales méthodes de prétraitement d'intensité
3. **Analyser** les histogrammes d'images et leur distribution
4. **Comparer** les performances des différentes stratégies :
   - Normalisation Min-Max
   - Égalisation globale (GHE)
   - Égalisation adaptative (CLAHE)
5. **Interpréter** les résultats sur des images réelles

---

## 📋 Prérequis

### Connaissances théoriques
- Notions de base en traitement d'images
- Compréhension des histogrammes
- Notions élémentaires de Python

### Logiciels requis
- Python 3.8 ou supérieur
- Un éditeur de code (VS Code, PyCharm, ou tout autre)

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
git clone https://github.com/votre-org/tp-vision-intensite.git
cd tp-vision-intensite
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

