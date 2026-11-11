import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from common import (ALPHA, BLUE, CONF, INK, MUTED, N_SIMS, ORANGE, SOURCE, fmt, h1_true, hypotheses,
                    load_population, load_sample, pct, run, simulate, true_value)

pop, smp = load_population(), load_sample()
N, n = len(pop), len(smp)
HS = hypotheses()
SIGN = {"less": "<", "greater": ">", "two-sided": r"\neq"}


def show(h, v):
    """Value as % for proportions, points for means."""
    return pct(v) if h["kind"] == "prop" else fmt(v)


def show_diff(h, v):
    """Difference: percentage points for proportions, points for means."""
    return f"{fmt(100 * v)} p.p." if h["kind"] == "prop" else f"{fmt(v)} pontos"


def n_for_power(h, true, sd=None, power=0.8):
    """Sample size that detects the real difference with the given power (normal approximation)."""
    za = stats.norm.ppf(1 - ALPHA / (2 if h["alternative"] == "two-sided" else 1))
    zb = stats.norm.ppf(power)
    if h["kind"] == "prop":
        return ((za * np.sqrt(h["h0"] * (1 - h["h0"])) + zb * np.sqrt(true * (1 - true))) / abs(true - h["h0"])) ** 2
    return ((za + zb) * sd / abs(true - h["h0"])) ** 2


def latex_num(h, v):
    return (f"{v:.3f}" if h["kind"] == "prop" else f"{v:.1f}").replace(".", "{,}")


def truth_chart(h, r, true):
    """CI from the sample (orange) against the true RS value and the H0 value, on one axis."""
    sym = "π" if h["kind"] == "prop" else "μ"
    names = ["Valor real (população RS)", f"H₀: {sym}₀ (Brasil)"]
    marks = pd.DataFrame({"Marca": names, "v": [true, h["h0"]], "y": -1, "y2": 1})
    ci = pd.DataFrame({"lo": [r["lo"]], "hi": [r["hi"]], "est": [r["est"]], "y": [0]})
    fmt_ = ".1%" if h["kind"] == "prop" else ".1f"
    x = alt.X("lo:Q", title=None, scale=alt.Scale(zero=False, padding=24),
              axis=alt.Axis(format="%" if h["kind"] == "prop" else "d", tickCount=8, grid=False))
    # dummy y scale: the CI sits at 0, the reference rules span the whole height
    y = alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-1, 1]))
    band = alt.Chart(ci).mark_rule(color=ORANGE, strokeWidth=6, strokeCap="round").encode(
        x, x2="hi:Q", y=y,
        tooltip=[alt.Tooltip("lo:Q", format=fmt_, title="Limite inferior"),
                 alt.Tooltip("hi:Q", format=fmt_, title="Limite superior")])
    point = alt.Chart(ci).mark_point(color=ORANGE, filled=True, size=140, stroke="white", strokeWidth=2).encode(
        alt.X("est:Q"), y=y, tooltip=[alt.Tooltip("est:Q", format=fmt_, title="Estimativa (amostra)")])
    legend = alt.Legend(orient="bottom", title=None)
    rules = alt.Chart(marks).mark_rule(strokeWidth=2).encode(
        alt.X("v:Q"), y, y2="y2:Q",
        color=alt.Color("Marca:N", scale=alt.Scale(domain=names, range=[BLUE, INK]), legend=legend),
        strokeDash=alt.StrokeDash("Marca:N", scale=alt.Scale(domain=names, range=[[1, 0], [5, 3]]), legend=legend),
        tooltip=["Marca:N", alt.Tooltip("v:Q", format=fmt_, title="Valor")])
    return (band + rules + point).properties(height=130)  # Streamlit counts axis + legend in this height


st.title("Painel 3: inferência estatística e comparação")
st.markdown(f"""
Cada hipótese compara um **parâmetro do RS** com o **valor do Brasil inteiro** (H₀). A inferência usa
**apenas a amostra** (n = {n}); depois conferimos o resultado com o valor real, calculado com os
{fmt(N, 0)} inscritos da população. Nível de significância **α = {fmt(ALPHA, 2)}** e intervalos de confiança de **{CONF:.0%}**
(bilaterais), fixados antes de olhar os dados.
""")

sims = simulate()
truths = {h["id"]: true_value(h, h["values"](pop)) for h in HS}
sims["contains"] = (sims["lo"] <= sims["id"].map(truths)) & (sims["id"].map(truths) <= sims["hi"])
agg = sims.groupby("id", sort=False).agg(reject=("p", lambda p: (p < ALPHA).mean()),
                                         cover=("contains", "mean"), se=("est", "std"))
A = fmt(ALPHA, 2)

summary = []
for h in HS:
    arr_s, arr_p = h["values"](smp), h["values"](pop)
    r, true = run(h, arr_s), true_value(h, arr_p)
    reject, h1 = r["p"] < ALPHA, h1_true(h, true)
    inside = r["lo"] <= true <= r["hi"]
    sym = r"\pi" if h["kind"] == "prop" else r"\mu"
    test = "Teste z para uma proporção" if h["kind"] == "prop" else "Teste t para uma média (σ desconhecido)"
    with st.container(border=True):
        st.subheader(f"{h['id']}: {h['question']}")
        st.markdown(f"**Parâmetro:** {h['param']}. **{test}.**")
        st.latex(rf"H_0: {sym} = {latex_num(h, h['h0'])} \quad\text{{vs}}\quad H_1: {sym} {SIGN[h['alternative']]} {latex_num(h, h['h0'])}")

        if h["kind"] == "prop":
            conds = [
                (True, "Amostra aleatória simples, observações independentes"),
                (n < 0.1 * N, f"n = {n} < 10% de N ({fmt(0.1 * N, 0)})"),
                (n * h["h0"] >= 10 and n * (1 - h["h0"]) >= 10,
                 f"Sucesso-fracasso sob H₀: nπ₀ = {fmt(n * h['h0'])} e n(1 − π₀) = {fmt(n * (1 - h['h0']))}, ambos ≥ 10"),
                (r["k"] >= 10 and n - r["k"] >= 10,
                 f"Sucesso-fracasso para o IC: {r['k']} sucessos e {n - r['k']} fracassos, ambos ≥ 10"),
            ]
        else:
            skew = pd.Series(arr_s).dropna().skew()
            conds = [
                (True, "Amostra aleatória simples, observações independentes"),
                (r["n"] < 0.1 * N, f"n = {r['n']} < 10% de N"),
                (r["n"] >= 30, f"n = {r['n']} ≥ 30 e assimetria da amostra = {fmt(skew, 2)}: pelo TCL, a média "
                               "amostral tem distribuição aproximadamente normal"),
                (True, "σ populacional desconhecido: usamos s e a distribuição t de Student "
                       f"com {r['n'] - 1} graus de liberdade"),
            ]
        with st.expander("Condições de aplicação", expanded=False):
            for ok, txt in conds:
                st.markdown(f"{'✅' if ok else '⚠️'} {txt}")

        c = st.columns([0.9, 1.8, 0.8, 1.1, 1.5])
        c[0].metric("Estimativa (amostra)", show(h, r["est"]))
        c[1].metric(f"IC {CONF:.0%}", f"{show(h, r['lo'])} a {show(h, r['hi'])}")
        c[2].metric("Estatística " + ("z" if h["kind"] == "prop" else "t"), fmt(r["stat"], 2))
        c[3].metric("Valor-p", "< 0,0001" if r["p"] < 1e-4 else fmt(r["p"], 4))
        c[4].metric("Decisão", "Rejeita H₀" if reject else "Não rejeita H₀")
        if reject:
            st.markdown(f"**Conclusão:** valor-p < α = {A}: rejeitamos H₀. A amostra traz evidência de que "
                        f"{h['h1_text']}.")
        else:
            st.markdown(f"**Conclusão:** valor-p ≥ α = {A}: não rejeitamos H₀. A amostra **não** traz evidência "
                        f"suficiente de que {h['h1_text']}.")

        st.markdown(f"**Confronto com a população:** valor real no RS = **{show(h, true)}** "
                    f"(estimativa {show(h, r['est'])}, erro amostral de {show_diff(h, r['est'] - true)}). "
                    f"O IC {'**contém**' if inside else '**não contém**'} o valor real. "
                    + ("A decisão está **correta**." if reject == h1 else
                       "A decisão é um **erro tipo II**: H₀ é falsa, mas a amostra não detectou a diferença."
                       if h1 else "A decisão é um **erro tipo I**."))
        st.altair_chart(truth_chart(h, r, true), width="stretch")
        st.caption("Barra laranja: IC 95% calculado com a amostra; ponto: estimativa pontual.")
        if not reject and h1:
            unit = "inscritos" if h["kind"] == "prop" else "notas desse grupo"
            power = agg.loc[h["id"], "reject"]
            why = (f"A diferença real (RS − Brasil) é de {show_diff(h, true - h['h0'])}; repetindo o sorteio "
                   f"{fmt(N_SIMS, 0)} vezes, **{pct(power, 0)}** das amostras rejeitam H₀ (o poder do teste). ")
            if power < 0.5:
                why += (f"Com {fmt(r['n'], 0)} {unit}, a amostra era pequena demais para essa diferença: para 80% de "
                        f"poder seriam necessários cerca de **{fmt(n_for_power(h, true, r.get('sd')), 0)} {unit}**.")
            else:
                why += (f"O teste tinha bom poder; esta amostra caiu nos {pct(1 - power, 0)} de sorteios em que a "
                        f"diferença passa despercebida (o erro amostral, de {show_diff(h, r['est'] - true)}, "
                        "puxou a estimativa na direção de H₀).")
            st.info("**Por que o teste não detectou?** " + why)

    summary.append({
        "Hipótese": h["id"], "Valor real (população)": show(h, true), "H₀ (Brasil)": show(h, h["h0"]),
        "Estimativa (amostra)": show(h, r["est"]), f"IC {CONF:.0%}": f"{show(h, r['lo'])} a {show(h, r['hi'])}",
        "IC contém o real?": "Sim" if inside else "Não", "Valor-p": "< 0,0001" if r["p"] < 1e-4 else fmt(r["p"], 4),
        "Decisão": "Rejeita H₀" if reject else "Não rejeita H₀",
        "H₀ é de fato falsa?": "Sim" if h1 else "Não",
        "Decisão correta?": "Sim" if reject == h1 else ("Não: erro tipo II" if h1 else "Não: erro tipo I"),
    })

st.header("Comparação: população × amostra")
st.dataframe(pd.DataFrame(summary).set_index("Hipótese"), width="stretch")

st.header("E se tivéssemos sorteado outra amostra?")
st.markdown(f"Repetimos o procedimento inteiro em **{fmt(N_SIMS, 0)} novas amostras** aleatórias de {n} inscritos. "
            "Isso mostra o quanto o resultado depende da sorte do sorteio.")
hs = {h["id"]: h for h in HS}
st.dataframe(pd.DataFrame({
    "Amostras que rejeitam H₀ (poder)": agg["reject"].map(pct),
    f"IC que contêm o valor real (esperado: {CONF:.0%})": agg["cover"].map(pct),
    "Desvio-padrão das estimativas (erro padrão)": [show(hs[i], v) for i, v in agg["se"].items()],
}).rename_axis("Hipótese"), width="stretch")

pick = st.selectbox("Ver 100 intervalos de confiança da hipótese", list(hs),
                    format_func=lambda i: f"{i}: {hs[i]['question']}")
h = hs[pick]
d = sims[(sims["id"] == pick) & (sims["sim"] < 100)].assign(
    IC=lambda x: x["contains"].map({True: "Contém o valor real", False: "Não contém"}))
fmt_ = ".1%" if h["kind"] == "prop" else ".1f"
base = alt.Chart(d).encode(alt.Y("sim:O", axis=None, title=None))
ci = base.mark_rule(strokeWidth=2).encode(
    alt.X("lo:Q", scale=alt.Scale(zero=False), title=None, axis=alt.Axis(format="%" if h["kind"] == "prop" else "d")),
    x2="hi:Q",
    color=alt.Color("IC:N", scale=alt.Scale(domain=["Contém o valor real", "Não contém"], range=[BLUE, ORANGE]),
                    legend=alt.Legend(orient="top", title=None)),
    tooltip=[alt.Tooltip("est:Q", format=fmt_, title="Estimativa"), alt.Tooltip("lo:Q", format=fmt_, title="Inferior"),
             alt.Tooltip("hi:Q", format=fmt_, title="Superior")])
lines = alt.Chart(pd.DataFrame({"Marca": ["Valor real (população RS)", "H₀ (Brasil)"],
                                "v": [truths[pick], h["h0"]]})).mark_rule(color=INK, strokeWidth=1.5).encode(
    x="v:Q", strokeDash=alt.StrokeDash("Marca:N", scale=alt.Scale(range=[[1, 0], [5, 3]]),
                                       legend=alt.Legend(orient="top", title=None)))
st.altair_chart((ci + lines).properties(height=520), width="stretch")
miss = int((~d["contains"]).sum())
st.caption(f"{100 - miss} dos 100 intervalos contêm o valor real; espera-se cerca de 95. A linha tracejada é H₀: "
           "quando ela fica fora do intervalo, aquela amostra rejeitaria H₀ num teste bilateral.")

st.header("Limitações")
st.markdown(f"""
- **Variabilidade amostral:** outra semente gera outra amostra e outros números. Com n = {n}, a chance de detectar
  a diferença real foi de {", ".join(f"{i} {pct(v, 0)}" for i, v in agg["reject"].items())}: só P1 é detectada
  em qualquer sorteio; nas outras, a decisão depende da sorte da amostra.
- **Tamanho da amostra:** n = {n} foi dimensionado para estimar proporções com margem de 5 pontos, não para detectar
  diferenças de 1 ponto percentual ou de 10 pontos na Redação. Não rejeitar H₀ não prova que H₀ é verdadeira.
- **Três hipóteses unilaterais:** P1, M1 e M2 testam uma direção definida antes de ver os dados; um IC bilateral de
  95% e um teste unilateral a 5% podem discordar quando o valor-p fica entre 0,025 e 0,05.
- **Subgrupos encolhem a amostra:** M1 usa só os sorteados sem computador **e** presentes, bem menos que {n},
  o que alarga o intervalo.
- **Definição da população:** o RS é definido pelo local da prova; quem mora no RS e fez a prova em outro estado
  fica de fora. Notas só existem para quem esteve presente.
- **Associação não é causa:** ter computador em casa anda junto com renda e tipo de escola; a diferença nas notas
  não mede o efeito do computador sozinho.
- **Questionário autodeclarado:** Q024 e Q025 são respostas do próprio inscrito.
""")
st.caption(SOURCE)
