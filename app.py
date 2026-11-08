import streamlit as st

st.set_page_config(page_title="Enem 2023 no RS: exclusão digital", page_icon="📊", layout="wide")

st.navigation([
    st.Page("panels/about.py", title="Base de dados e amostra", default=True),
    st.Page("panels/population.py", title="Painel 1: população", url_path="populacao"),
    st.Page("panels/sample.py", title="Painel 2: amostra", url_path="amostra"),
    st.Page("panels/inference.py", title="Painel 3: inferência", url_path="inferencia"),
]).run()

st.sidebar.caption("IA003 · Estatística para IA · UFRGS\n\nAndré Bergmann Lisboa")
