# Assistant pédagogique en géomatique – Université de Montréal

Ce projet Streamlit met en place un **assistant conversationnel intelligent** pour les étudiants de géographie et de géomatique (GEO1532 & GEO2512).  
Il s’appuie sur les contenus de cours, travaux pratiques et le manuel *GIS Fundamentals (Bolstad, 5e éd.)*.

---

## 🚀 Installation locale

1️⃣ **Installer les dépendances**
```bash
pip install -r requirements.txt
```

2️⃣ **Créer le dossier des documents**
Placez vos fichiers (cours, TPs, livre) dans un dossier `./docs/` :
```
./docs/
├── GIS_Fundamentals.pdf
├── TP1_JDaoustFGirard.docx
├── TP2_JDaoustFGirard.docx
└── autres_fichiers...
```

3️⃣ **Définir la clé OpenAI**
```bash
export OPENAI_API_KEY="votre_cle"
```

4️⃣ **Lancer le chatbot**
```bash
streamlit run chatbot_geomatique.py
```

5️⃣ **Ouvrir le navigateur**
> http://localhost:8501

---

## ☁️ Déploiement sur Streamlit Cloud

1️⃣ Créez un dépôt GitHub avec :
- `chatbot_geomatique.py`
- `requirements.txt`
- Dossier `./docs/`

2️⃣ Connectez [Streamlit Cloud](https://share.streamlit.io) à votre dépôt.

3️⃣ Dans la section **Secrets**, ajoutez :
```
OPENAI_API_KEY = "votre_cle"
```

4️⃣ Déployez : vous obtiendrez un lien du type :
> https://geomatique-udem.streamlit.app

---

## 🧭 Utilisation

- **Mode conversationnel** : poser des questions sur la géomatique (cartographie, SIG, GNSS, projections…).  
- **Mode administrateur** : corriger les réponses ou annoter le contenu.  
- **Mode quiz** : tester ses connaissances (50 questions couvrant GEO1532 et GEO2512).

---

## 👨‍🏫 Auteur

Créé par **François Girard**  
Département de géographie – Université de Montréal (2025)

---
