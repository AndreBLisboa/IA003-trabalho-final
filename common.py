"""Shared data, sampling, statistics and charts for the three panels.

Code is in English; everything shown on screen is in Portuguese (the course language).
"""
from math import ceil
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

DATA = Path(__file__).parent / "data"
SEED = 2026  # fixed seed: the same simple random sample on every run
CONF = 0.95
ALPHA = 0.05
MARGIN = 0.05  # margin of error used to size the sample
P_PLAN = 0.5  # worst case p(1-p), no prior guess needed
N_SIMS = 1000

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
SOURCE = (
    "Fonte: INEP, [Microdados do Enem 2023]"
    "(https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem)."
)

# --- labels (Portuguese, shown on screen) ------------------------------------
COMPUTER = {"A": "Nenhum", "B": "1", "C": "2", "D": "3", "E": "4 ou mais"}
INTERNET = {"A": "Não", "B": "Sim"}
PRESENCE = {0: "Faltou", 1: "Presente", 2: "Eliminado"}
SCHOOL = {1: "Não respondeu", 2: "Pública", 3: "Privada"}
SEX = {"F": "Feminino", "M": "Masculino"}
# Q006 letters grouped into minimum-wage bands (2023 minimum wage = R$ 1.320)
INCOME = {"A": "Sem renda", "B": "Até 1 SM"} | dict.fromkeys("CD", "1 a 2 SM") \
    | dict.fromkeys("EFG", "2 a 4 SM") | dict.fromkeys("HI", "4 a 6 SM") \
    | dict.fromkeys("JKLMNOPQ", "Mais de 6 SM")

CAT_VARS = {  # column -> (label, category order)
    "computer": ("Computadores na residência (Q024)", list(COMPUTER.values())),
    "internet": ("Acesso à internet na residência (Q025)", list(INTERNET.values())),
    "presence_mt": ("Presença na prova de Matemática", list(PRESENCE.values())),
    "school": ("Tipo de escola do ensino médio", list(SCHOOL.values())),
    "income": ("Renda mensal da família (Q006)", list(dict.fromkeys(INCOME.values()))),
    "sex": ("Sexo", list(SEX.values())),
}
NUM_VARS = {
    "score_mt": "Nota de Matemática",
    "score_essay": "Nota da Redação",
    "score_cn": "Nota de Ciências da Natureza",
    "score_ch": "Nota de Ciências Humanas",
    "score_lc": "Nota de Linguagens e Códigos",
}


def fmt(x, d=1):
    """pt-BR number: 1.234,5"""
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x, d=1):
    return fmt(100 * x, d) + "%"


# --- data --------------------------------------------------------------------
@st.cache_data
def load_population():
    raw = pd.read_parquet(DATA / "population_rs.parquet")
    df = pd.DataFrame({
        "id": raw["NU_INSCRICAO"],
        "computer": raw["Q024"].map(COMPUTER),
        "internet": raw["Q025"].map(INTERNET),
        "presence_mt": raw["TP_PRESENCA_MT"].map(PRESENCE),
        "school": raw["TP_ESCOLA"].map(SCHOOL),
        "income": raw["Q006"].map(INCOME),
        "sex": raw["TP_SEXO"].map(SEX),
        "score_mt": raw["NU_NOTA_MT"],
        "score_essay": raw["NU_NOTA_REDACAO"],
        "score_cn": raw["NU_NOTA_CN"],
        "score_ch": raw["NU_NOTA_CH"],
        "score_lc": raw["NU_NOTA_LC"],
    })
    for col, (_, order) in CAT_VARS.items():
        df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    return df


@st.cache_data
def load_reference():
    return pd.read_csv(DATA / "brazil_reference.csv", index_col=0)["brazil"]


def sample_size(N, conf=CONF, e=MARGIN, p=P_PLAN):
    """Sample size for a proportion, with finite-population correction (same as the USP/Bauru calculator)."""
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    n0 = z**2 * p * (1 - p) / e**2
    return ceil(n0 / (1 + (n0 - 1) / N)), n0, z


@st.cache_data
def load_sample():
    """Simple random sample without replacement: every candidate has probability n/N."""
    pop = load_population()
    n, _, _ = sample_size(len(pop))
    return pop.sample(n=n, random_state=SEED)


# --- descriptive statistics --------------------------------------------------
def describe(df):
    rows = {}
    for col, label in NUM_VARS.items():
        s = df[col].dropna()
        rows[label] = {
            "n": len(s), "Média": s.mean(), "Mediana": s.median(), "Desvio-padrão": s.std(),
            "Mín": s.min(), "Q1": s.quantile(.25), "Q3": s.quantile(.75), "Máx": s.max(),
            "Assimetria": s.skew(),
        }
    return pd.DataFrame(rows).T


def frequencies(df, col):
    f = df[col].value_counts(sort=False).rename("Frequência").to_frame()
    f["Proporção"] = f["Frequência"] / f["Frequência"].sum()
    f.index.name = CAT_VARS[col][0]
    return f


# --- charts ------------------------------------------------------------------
def hist_chart(df, col, color):
    s = df[[col]].dropna()
    base = alt.Chart(s)
    bars = base.mark_bar(color=color, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        alt.X(f"{col}:Q", bin=alt.Bin(maxbins=40), title=NUM_VARS[col]),
        alt.Y("count():Q", title="Frequência"),
        tooltip=[alt.Tooltip(f"{col}:Q", bin=alt.Bin(maxbins=40), title="Faixa"),
                 alt.Tooltip("count():Q", title="Frequência")],
    )
    lines = pd.DataFrame({"Medida": ["Média", "Mediana"], "v": [s[col].mean(), s[col].median()]})
    rules = alt.Chart(lines).mark_rule(strokeWidth=2).encode(
        x="v:Q",
        strokeDash=alt.StrokeDash("Medida:N", scale=alt.Scale(range=[[1, 0], [5, 3]]), title=None),
        color=alt.value(INK),
        tooltip=["Medida", alt.Tooltip("v:Q", format=".1f", title="Valor")],
    )
    return (bars + rules).properties(height=300)


def box_by_chart(df, num_col, cat_col, color):
    d = df[[num_col, cat_col]].dropna()
    return alt.Chart(d).mark_boxplot(color=color, size=28, outliers={"size": 6, "opacity": .3}).encode(
        alt.X(f"{num_col}:Q", title=NUM_VARS[num_col], scale=alt.Scale(zero=False)),
        alt.Y(f"{cat_col}:N", sort=CAT_VARS[cat_col][1], title=CAT_VARS[cat_col][0]),
    ).properties(height=260)


def prop_chart(df, col, color):
    f = frequencies(df, col).reset_index().rename(columns={CAT_VARS[col][0]: "cat"})
    base = alt.Chart(f).encode(alt.Y("cat:N", sort=CAT_VARS[col][1], title=None))
    bars = base.mark_bar(color=color, cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
        alt.X("Proporção:Q", axis=alt.Axis(format="%"), title="Proporção"),
        tooltip=[alt.Tooltip("cat:N", title="Categoria"), "Frequência:Q",
                 alt.Tooltip("Proporção:Q", format=".1%")],
    )
    labels = base.mark_text(align="left", dx=4, color=MUTED).encode(
        alt.X("Proporção:Q"), alt.Text("Proporção:Q", format=".1%"))
    return (bars + labels).properties(height=40 + 30 * len(f))


def compare_prop_chart(pop, smp, col):
    f = pd.concat([
        frequencies(pop, col).assign(Base="População"),
        frequencies(smp, col).assign(Base="Amostra"),
    ]).reset_index().rename(columns={CAT_VARS[col][0]: "cat"})
    return alt.Chart(f).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
        alt.Y("cat:N", sort=CAT_VARS[col][1], title=None),
        alt.YOffset("Base:N", sort=["População", "Amostra"]),
        alt.X("Proporção:Q", axis=alt.Axis(format="%")),
        alt.Color("Base:N", scale=alt.Scale(domain=["População", "Amostra"], range=[BLUE, ORANGE]),
                  legend=alt.Legend(orient="top", title=None)),
        tooltip=["Base:N", alt.Tooltip("cat:N", title="Categoria"), "Frequência:Q",
                 alt.Tooltip("Proporção:Q", format=".1%")],
    ).properties(height=60 + 44 * f["cat"].nunique())


def descriptive_panel(df, color, who):
    """Panel 1 and 2 share this body: same indicators and charts, on the population or on the sample."""
    c = st.columns(4)
    c[0].metric("Inscritos", fmt(len(df), 0))
    c[1].metric("Sem computador em casa", pct((df["computer"] == "Nenhum").mean()))
    c[2].metric("Faltaram à prova de Matemática", pct((df["presence_mt"] == "Faltou").mean()))
    c[3].metric("Média em Matemática", fmt(df["score_mt"].mean()))

    st.subheader("Variáveis quantitativas: as notas")
    st.dataframe(describe(df).style.format(lambda v: fmt(v, 1), subset=pd.IndexSlice[:, "Média":])
                 .format(lambda v: fmt(v, 0), subset=["n"]), width="stretch")
    st.caption("Desvio-padrão amostral (divisor n − 1). Notas existem só para quem esteve presente na prova.")
    num = st.selectbox("Nota", list(NUM_VARS), format_func=NUM_VARS.get, key=f"num_{who}")
    left, right = st.columns(2)
    left.markdown(f"**Distribuição: {NUM_VARS[num]}**")
    left.altair_chart(hist_chart(df, num, color), width="stretch")
    right.markdown(f"**{NUM_VARS[num]} por número de computadores em casa**")
    right.altair_chart(box_by_chart(df, num, "computer", color), width="stretch")

    st.subheader("Variáveis qualitativas: frequências e proporções")
    cat = st.selectbox("Variável", list(CAT_VARS), format_func=lambda k: CAT_VARS[k][0], key=f"cat_{who}")
    left, right = st.columns([1, 1])
    left.altair_chart(prop_chart(df, cat, color), width="stretch")
    right.dataframe(frequencies(df, cat).style.format({"Frequência": lambda v: fmt(v, 0), "Proporção": lambda v: pct(v)}), width="stretch")


# --- inference ---------------------------------------------------------------
def p_value(stat, alternative, dist):
    if alternative == "less":
        return dist.cdf(stat)
    if alternative == "greater":
        return dist.sf(stat)
    return 2 * dist.sf(abs(stat))


def prop_test(k, n, pi0, alternative, conf=CONF):
    """One-proportion z test (SE under H0) and Wald confidence interval (SE from the sample), as in class."""
    p = k / n
    z = (p - pi0) / np.sqrt(pi0 * (1 - pi0) / n)
    zc = stats.norm.ppf(1 - (1 - conf) / 2)
    me = zc * np.sqrt(p * (1 - p) / n)
    return dict(n=n, k=k, est=p, lo=p - me, hi=p + me, me=me, stat=z,
                p=p_value(z, alternative, stats.norm), crit=zc)


def mean_test(x, mu0, alternative, conf=CONF):
    """One-sample t test (sigma unknown) and t confidence interval."""
    n, m, s = len(x), x.mean(), x.std(ddof=1)
    se = s / np.sqrt(n)
    t = (m - mu0) / se
    tc = stats.t.ppf(1 - (1 - conf) / 2, df=n - 1)
    return dict(n=n, est=m, sd=s, lo=m - tc * se, hi=m + tc * se, me=tc * se, stat=t,
                p=p_value(t, alternative, stats.t(df=n - 1)), crit=tc)


def hypotheses():
    """The four hypotheses. H0 values come from all of Brazil; the parameter is for RS.

    `values(df)` returns a boolean array (event, for proportions) or a float array with NaN
    outside the subgroup (for means), aligned with df's rows.
    """
    ref = load_reference()
    return [
        dict(id="P1", kind="prop", alternative="less", h0=ref["prop_no_computer"],
             param="π = proporção de inscritos do RS **sem computador** em casa",
             question="O RS tem menos inscritos sem computador em casa do que o Brasil?",
             h1_text="a proporção de inscritos sem computador em casa no RS é menor que a do Brasil",
             values=lambda d: (d["computer"] == "Nenhum").to_numpy()),
        dict(id="P2", kind="prop", alternative="two-sided", h0=ref["prop_absent_mt"],
             param="π = proporção de inscritos do RS que **faltaram** à prova de Matemática",
             question="A abstenção no RS é diferente da do Brasil?",
             h1_text="a abstenção na prova de Matemática no RS é diferente da do Brasil",
             values=lambda d: (d["presence_mt"] == "Faltou").to_numpy()),
        dict(id="M1", kind="mean", alternative="less", h0=ref["mean_score_mt"],
             param="μ = nota média em Matemática dos inscritos do RS **sem computador** em casa (presentes)",
             question="Quem não tem computador em casa fica abaixo da média nacional em Matemática?",
             h1_text="os inscritos do RS sem computador em casa têm média em Matemática abaixo da média nacional",
             values=lambda d: d["score_mt"].where(d["computer"] == "Nenhum").to_numpy()),
        dict(id="M2", kind="mean", alternative="greater", h0=ref["mean_essay"],
             param="μ = nota média na Redação dos inscritos do RS (presentes)",
             question="O RS tem nota média na Redação maior que a do Brasil?",
             h1_text="a nota média na Redação no RS é maior que a do Brasil",
             values=lambda d: d["score_essay"].to_numpy()),
    ]


def run(h, arr):
    if h["kind"] == "prop":
        return prop_test(int(arr.sum()), len(arr), h["h0"], h["alternative"])
    return mean_test(arr[~np.isnan(arr)], h["h0"], h["alternative"])


def true_value(h, arr):
    return arr.mean() if h["kind"] == "prop" else np.nanmean(arr)


def h1_true(h, true):
    return {"less": true < h["h0"], "greater": true > h["h0"], "two-sided": true != h["h0"]}[h["alternative"]]


@st.cache_data
def simulate(n_sims=N_SIMS):
    """Repeat the whole procedure on n_sims new samples of the same size, to see sampling variability."""
    pop = load_population()
    n = len(load_sample())
    rng = np.random.default_rng(SEED)
    arrays = {h["id"]: (h, h["values"](pop)) for h in hypotheses()}
    rows = []
    for i in range(n_sims):
        idx = rng.choice(len(pop), size=n, replace=False)
        for hid, (h, arr) in arrays.items():
            r = run(h, arr[idx])
            rows.append(dict(sim=i, id=hid, est=r["est"], lo=r["lo"], hi=r["hi"], p=r["p"]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # self-check: formulas against statsmodels/scipy and the textbook sample size
    from statsmodels.stats.proportion import proportions_ztest
    assert sample_size(10**9)[0] == 385 and round(sample_size(10**9)[1], 2) == 384.15
    r = prop_test(30, 100, .4, "less")
    z_sm, p_sm = proportions_ztest(30, 100, .4, alternative="smaller", prop_var=.4)
    assert np.isclose(r["stat"], z_sm) and np.isclose(r["p"], p_sm)
    x = np.random.default_rng(0).normal(10, 2, 50)
    r = mean_test(x, 9.5, "greater")
    t_sp = stats.ttest_1samp(x, 9.5, alternative="greater")
    assert np.isclose(r["stat"], t_sp.statistic) and np.isclose(r["p"], t_sp.pvalue)
    print("ok")
