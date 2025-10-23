import streamlit as st, importlib, glob
st.title("Healthcheck – Chatbot Géomatique")
for m in ["openai","tiktoken","numpy","pypdf","pptx","docx","streamlit"]:
    try:
        importlib.import_module(m)
        st.success(f"OK: {m}")
    except Exception as e:
        st.error(f"KO: {m} -> {e}")
st.write("Contenu ./docs:", glob.glob("docs/*"))
