import pandas as pd
import streamlit as st

from common import (CAT_VARS, ORANGE, SEED, SOURCE, compare_prop_chart, describe,
                    descriptive_panel, fmt, load_population, load_sample, pct)

pop, smp = load_population(), load_sample()

st.title("Painel 2: análise descritiva da amostra")
st.markdown(f"**Amostra aleatória simples** de {len(smp)} inscritos (semente {SEED}). "
            "Todos os números desta página usam só a amostra.")
descriptive_panel(smp, ORANGE, "smp")

st.header("Amostra × população")
st.markdown("As diferenças abaixo são **erro amostral**: a amostra é aleatória, então não acerta "
            "exatamente os valores da população, mas fica perto deles.")
d = pd.DataFrame({"População": describe(pop)["Média"], "Amostra": describe(smp)["Média"]})
d["Diferença"] = d["Amostra"] - d["População"]
d["Diferença (%)"] = d["Diferença"] / d["População"]
st.markdown("**Média das notas**")
st.dataframe(d.style.format({c: lambda v: fmt(v) for c in ["População", "Amostra", "Diferença"]}
                              | {"Diferença (%)": lambda v: pct(v)}), width="stretch")
cat = st.selectbox("Proporções por categoria", list(CAT_VARS), format_func=lambda k: CAT_VARS[k][0])
st.altair_chart(compare_prop_chart(pop, smp, cat), width="stretch")
st.caption(SOURCE)
