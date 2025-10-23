# -*- coding: utf-8 -*-
"""
Assistant pédagogique en géomatique – Université de Montréal
Streamlit chatbot pour les cours GEO1532 (1re année) et GEO2512 (2e année)
Charge les contenus depuis ./docs/ (PDF, DOCX, PPTX) et répond aux questions.
Inclut : Mode administrateur (corrections) et Mode quiz (50 questions).
"""

import os
import random
import streamlit as st

# --- Imports LangChain (compatibles avec versions récentes) ---
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except Exception:
    # Compat rétro si ancien LangChain
    from langchain.chat_models import ChatOpenAI  # type: ignore
    from langchain.embeddings import OpenAIEmbeddings  # type: ignore

try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import (
        PyPDFLoader,
        UnstructuredWordDocumentLoader,
        UnstructuredPowerPointLoader,
    )
except Exception:
    # Compat rétro
    from langchain.vectorstores import FAISS  # type: ignore
    from langchain.document_loaders import (  # type: ignore
        PyPDFLoader,
        UnstructuredWordDocumentLoader,
        UnstructuredPowerPointLoader,
    )

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain


# ---------------------------
# CONFIGURATION
# ---------------------------

st.set_page_config(page_title="Assistant Géomatique UdeM", page_icon="🗺️", layout="wide")

st.title("🗺️ Assistant pédagogique en géomatique – Université de Montréal")
st.markdown(
    """
Cet assistant aide les étudiants à comprendre les concepts de **cartographie**, **SIG**, **GNSS**,
**projections** et **analyses spatiales**.

**Sources intégrées** (déposez vos fichiers dans `./docs/`) :
- Cours 1re et 2e année (PowerPoint)  
- Travaux pratiques (GEO1532 & GEO2512)  
- Livre *GIS Fundamentals* (Bolstad, 5e éd.)

Posez une question : *« Quelle est la différence entre raster et vecteur ? »*, *« Comment géoréférencer une carte ? »*, etc.
"""
)

if os.getenv("OPENAI_API_KEY") in (None, "", "changeme"):
    st.warning(
        "⚠️ La variable d'environnement **OPENAI_API_KEY** n'est pas définie. "
        "Ajoutez votre clé pour activer le moteur de réponses."
    )

# ---------------------------
# CHARGEMENT DES DOCUMENTS (dossier ./docs)
# ---------------------------

@st.cache_resource(show_spinner=False)
def load_docs():
    docs = []
    base_path = os.path.join(os.getcwd(), "docs")

    if not os.path.exists(base_path):
        st.error(
            f"Le dossier `{base_path}` est introuvable. "
            "Créez un dossier `./docs` et placez-y vos fichiers (.pdf, .docx, .pptx)."
        )
        return []

    files = [f for f in os.listdir(base_path) if f.lower().endswith((".pdf", ".docx", ".pptx"))]
    if not files:
        st.warning("Aucun fichier détecté dans `./docs`. "
                   "Veuillez y placer vos documents de cours puis recharger l'application.")
        return []

    for f in files:
        path = os.path.join(base_path, f)
        try:
            if f.lower().endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif f.lower().endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(path)
            elif f.lower().endswith(".pptx"):
                loader = UnstructuredPowerPointLoader(path)
            else:
                continue
            docs.extend(loader.load())
        except Exception as e:
            st.warning(f"Erreur lors du chargement de `{f}` : {e}")
    return docs


documents = load_docs()
if not documents:
    st.stop()

# Découpage pour index (chunks)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
texts = text_splitter.split_documents(documents)

# ---------------------------
# CRÉATION DE L'INDEX VECTORIEL
# ---------------------------

@st.cache_resource(show_spinner=False)
def create_vectorstore(texts):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

vectorstore = create_vectorstore(texts)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ---------------------------
# CHAÎNE CONVERSATIONNELLE
# ---------------------------

# Modèle raisonnable côté coût/latence, FR par défaut.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

# ---------------------------
# ÉTAT & BARRE LATÉRALE
# ---------------------------

if "history" not in st.session_state:
    st.session_state["history"] = []

st.sidebar.header("⚙️ Mode d’administration")
admin_mode = st.sidebar.checkbox("Activer le mode administrateur")
quiz_mode = st.sidebar.checkbox("Activer le mode quiz étudiant")

# ---------------------------
# MODE ADMINISTRATEUR
# ---------------------------

if admin_mode:
    st.markdown("### 🔧 Mode administrateur activé")
    st.info("Ajoutez/retirez des fichiers directement dans `./docs`. "
            "Vous pouvez aussi corriger une réponse via le formulaire ci-dessous.")

    st.sidebar.subheader("Corriger ou enrichir une réponse")
    correction_q = st.sidebar.text_area("Question à corriger")
    correction_a = st.sidebar.text_area("Nouvelle réponse ou note explicative")
    if st.sidebar.button("Enregistrer la correction"):
        st.session_state["history"].append((correction_q, f"[Révision enseignant] {correction_a}"))
        st.success("Révision ajoutée à l'historique.")

# ---------------------------
# MODE QUIZ – 50 QUESTIONS
# ---------------------------

QUIZ_QUESTIONS = [
    ("Différence principale entre modèle raster et vectoriel ?", "Raster: cellules; Vecteur: entités géométriques."),
    ("Projection qui conserve les angles mais déforme les surfaces ?", "Conforme (ex. Mercator)."),
    ("Projection qui conserve les surfaces mais déforme les formes ?", "Équivalente (ex. Albers)."),
    ("Méthode d'interpolation par moyenne pondérée par la distance ?", "IDW."),
    ("Système GNSS européen ?", "Galileo."),
    ("Rôle du géoréférencement ?", "Associer une image à des coordonnées réelles."),
    ("Différence MNA vs MNT ?", "MNA inclut objets; MNT = terrain nu."),
    ("Format couramment associé au vecteur ?", "Shapefile (.shp)."),
    ("Que mesure l’indicatrice de Tissot ?", "Déformations locales de projection."),
    ("Qu’est-ce qu’un buffer ?", "Zone tampon autour d’une entité."),
    ("Différence IDW vs krigeage ?", "Krigeage utilise un variogramme statistique."),
    ("Définition base de données relationnelle ?", "Tables reliées par clés primaires/étrangères."),
    ("Outil pour automatiser des traitements SIG ?", "Model Builder."),
    ("Projections usuelles au Québec (topo) ?", "UTM / MTM."),
    ("Déclinaison magnétique ?", "Angle entre nord géographique et nord magnétique."),
    ("GNSS signifie ?", "Global Navigation Satellite System."),
    ("Classification en intervalles égaux ?", "Égal-intervalle."),
    ("Principe de l’analyse multicritère ?", "Combinaison de couches pondérées."),
    ("Projection conique équivalente avec 2 parallèles standards ?", "Albers égale surface."),
    ("Différence GPS vs GNSS ?", "GPS = américain; GNSS = tous systèmes."),
    ("Analyse pour zones d’accessibilité ?", "Analyse de réseaux."),
    ("WGS84 ?", "Datum géodésique mondial (1984)."),
    ("Calcul de pente à partir d’un MNT ?", "Outil Slope."),
    ("Projection équivalente utilisée pour le Canada ?", "Albers égale surface."),
    ("Opération spatiale par intersection ?", "Intersect."),
    ("Donnée pour courbes de niveau (QC/CA) ?", "BNDT (ou équivalents)."),
    ("Estim. volume de carrière ?", "Différence entre MNT (Cut/Fill)."),
    ("Organigramme SIG ?", "Représentation des étapes d’un traitement."),
    ("Extraction zone par masque (raster) ?", "Extract by Mask."),
    ("Variable visuelle pour hiérarchie ?", "Taille / valeur (luminosité)."),
    ("Autocorrélation spatiale positive ?", "Valeurs similaires proches."),
    ("Formule DMS → degrés décimaux ?", "D + M/60 + S/3600."),
    ("Table attributaire ?", "Données descriptives des entités."),
    ("Échelle raisonnable carte Québec ?", "≈ 1:1 000 000."),
    ("Taille de pixel Landsat 8 ?", "30 m (bandes réflectives)."),
    ("Raster vs LiDAR ?", "Raster = grille; LiDAR = nuage de points XYZ."),
    ("Formats raster courants ?", ".tif, .img."),
    ("Variogramme, utilité ?", "Mesure dépendance spatiale."),
    ("RMSE ?", "Erreur quadratique moyenne."),
    ("Reclassification ?", "Transformation des valeurs d’un raster en classes."),
    ("Sélection attributaire dans QGIS ?", "Sélection par expression."),
    ("Jointure attributaire ?", "Relier table externe par clé commune."),
    ("Topologie (vecteur) ?", "Relations spatiales entre entités (adjacency/connectivité)."),
    ("SIG – composantes ?", "Logiciel, matériel, données, méthodes, utilisateurs."),
    ("Systèmes de coordonnées courants ?", "Géographiques (lat/long), projetés (UTM/MTM)."),
    ("Azimut ?", "Angle mesuré depuis le nord, dans le sens horaire."),
    ("Échantillonnage pour interpolation ?", "Représentatif, couvrant gradients & extrêmes."),
    ("LISA ?", "Indicateurs locaux d’association spatiale."),
    ("Moran I ?", "Mesure globale d’autocorrélation spatiale."),
]

if quiz_mode:
    st.header("🎯 Quiz interactif – Teste tes connaissances en géomatique !")
    question, expected = random.choice(QUIZ_QUESTIONS)
    st.subheader(question)
    answer = st.text_input("Ta réponse :")
    if st.button("Vérifier"):
        st.markdown(f"**Réponse attendue :** {expected}")
        st.success("Bien joué — continue tes apprentissages !")

# ---------------------------
# MODE CONVERSATIONNEL
# ---------------------------

if not quiz_mode:
    user_query = st.text_input("💬 Pose ta question sur la géomatique :")
    if user_query:
        with st.spinner("Réflexion en cours..."):
            response = qa_chain({"question": user_query, "chat_history": st.session_state["history"]})
            answer = response["answer"]
            st.session_state["history"].append((user_query, answer))
            st.markdown(f"### 🧠 Réponse\n{answer}")

# ---------------------------
# HISTORIQUE
# ---------------------------

if st.session_state["history"]:
    with st.expander("🕘 Historique des questions"):
        for q, a in reversed(st.session_state["history"]):
            st.markdown(f"**Q :** {q}\n\n**R :** {a}\n---")
