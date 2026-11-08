import streamlit as st

from common import CONF, MARGIN, P_PLAN, SEED, SOURCE, fmt, load_population, load_sample, sample_size

pop, smp = load_population(), load_sample()
N = len(pop)
n, n0, z = sample_size(N)

st.title("Enem 2023 no Rio Grande do Sul: a exclusão digital")
st.markdown(
    "**Pergunta:** quantos candidatos do RS ainda fazem o Enem sem computador em casa, e isso pesa no "
    "desempenho e na presença? Comparamos o RS com o Brasil usando **apenas uma amostra** e conferimos "
    "o resultado contra a população inteira, que conhecemos."
)

st.header("A base de dados")
st.markdown(f"""
- **Fonte:** Instituto Nacional de Estudos e Pesquisas Educacionais Anísio Teixeira (INEP), órgão do
  Ministério da Educação. [Microdados do Enem 2023](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem)
  ([download direto](https://download.inep.gov.br/microdados/microdados_enem_2023.zip)).
- **Contexto:** o Exame Nacional do Ensino Médio avalia quem termina o ensino médio e é a porta de entrada
  para universidades públicas (Sisu), Prouni e Fies. Os microdados trazem, para cada inscrito, as notas,
  a presença e as respostas ao questionário socioeconômico.
- **Tamanho:** 3.933.955 inscritos no Brasil e 76 variáveis.
- **População de interesse:** os **{fmt(N, 0)} inscritos** cuja prova seria aplicada no RS
  (`SG_UF_PROVA = RS`; a base de 2023 não traz o município de residência).
- **Unidade amostral:** o inscrito.
- **Brasil como referência:** os valores das hipóteses nulas são os parâmetros do Brasil inteiro,
  calculados com todos os 3,9 milhões de inscritos.
""")

st.subheader("Variáveis selecionadas")
st.markdown("""
| Variável | Código INEP | Tipo | Uso |
|---|---|---|---|
| Computadores na residência | Q024 | Qualitativa ordinal (nenhum, 1, 2, 3, 4 ou mais) | Proporção sem computador (P1); grupo de M1 |
| Acesso à internet na residência | Q025 | Qualitativa nominal (sim/não) | Descritiva |
| Presença na prova de Matemática | TP_PRESENCA_MT | Qualitativa nominal (faltou, presente, eliminado) | Proporção de faltosos (P2) |
| Tipo de escola, renda familiar, sexo | TP_ESCOLA, Q006, TP_SEXO | Qualitativas | Descritiva |
| Nota de Matemática | NU_NOTA_MT | Quantitativa contínua | Média (M1) |
| Nota da Redação | NU_NOTA_REDACAO | Quantitativa discreta (0 a 1000, passos de 20) | Média (M2) |
| Notas de Ciências da Natureza, Humanas e Linguagens | NU_NOTA_CN, _CH, _LC | Quantitativas contínuas | Descritiva |
""")

st.header("Dimensionamento da amostra")
st.markdown(f"""
O tamanho foi calculado para estimar uma **proporção**, como na
[calculadora da USP/Bauru](http://estatistica.bauru.usp.br/calculoamostral/ta_ic_proporcao.php)
([cópia no Internet Archive](https://web.archive.org/web/20261013000842/http://estatistica.bauru.usp.br/calculoamostral/ta_ic_proporcao.php)),
opção "Tamanho da amostra para intervalo de confiança de uma proporção":

- **Nível de confiança: {CONF:.0%}** (z = {fmt(z, 2)}), o padrão da área: aceitamos errar 5% das vezes.
- **Margem de erro: {MARGIN:.0%}** (5 pontos percentuais), precisão suficiente para as proporções estudadas,
  que vão de cerca de 30% a 50%.
- **Proporção planejada: p = {fmt(P_PLAN)}**, o pior caso: maximiza p(1 − p) e garante a margem para qualquer
  proporção, sem precisar de um palpite prévio.
- **Tamanho da população: N = {fmt(N, 0)}** (correção para população finita).
""")
st.latex(rf"n_0 = \frac{{z^2\,p(1-p)}}{{e^2}} = \frac{{{z:.2f}^2 \cdot 0{{,}}5 \cdot 0{{,}}5}}{{0{{,}}05^2}}"
         rf" = {n0:.2f}".replace(".", "{,}"))
st.latex(rf"n = \frac{{n_0}}{{1 + \frac{{n_0 - 1}}{{N}}}} = \frac{{{n0:.2f}}}{{1 + \frac{{{n0 - 1:.2f}}}{{{N}}}}}"
         rf" = {n0 / (1 + (n0 - 1) / N):.2f} \Rightarrow n = {n}".replace(".", "{,}"))
st.markdown(
    f"Como N é grande, a correção quase não muda o resultado ({fmt(n0, 0)} → {n}). "
    "Para as médias, a precisão obtida aparece no Painel 3 como a margem de erro de cada intervalo."
)

st.header("Seleção da amostra aleatória simples")
st.markdown(f"""
Sorteio **sem reposição** de {n} inscritos entre os {fmt(N, 0)}, com o gerador de números aleatórios do pandas
e **semente fixa {SEED}** (o mesmo sorteio em toda execução, portanto reprodutível):
""")
st.code(f"amostra = populacao.sample(n={n}, random_state={SEED})", language="python")
st.markdown(f"Cada inscrito tem a mesma probabilidade de ser sorteado: {n}/{fmt(N, 0)} = {fmt(100 * n / N, 2)}%. "
            "A amostra sorteada pode ser baixada abaixo.")
st.download_button("Baixar a amostra (CSV)", smp.to_csv(index=False).encode("utf-8"),
                   "amostra_enem2023_rs.csv", "text/csv")
st.caption(SOURCE)
