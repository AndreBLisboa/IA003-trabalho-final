import streamlit as st

from common import BLUE, SOURCE, descriptive_panel, load_population

st.title("Painel 1: análise descritiva da população")
st.markdown("**População:** todos os 159.919 inscritos no Enem 2023 que fariam a prova no Rio Grande do Sul.")
descriptive_panel(load_population(), BLUE, "pop")
st.caption(SOURCE)
