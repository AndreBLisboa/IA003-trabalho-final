# IA003 final assignment: ENEM 2023 in Rio Grande do Sul and the digital divide

Individual assignment for **IA003: Statistics for AI** (UFRGS, Prof. Lisiane Selau), by André Bergmann Lisboa.
It applies sampling, confidence intervals and hypothesis tests to a large dataset, comparing the inference
made from a simple random sample with the true values of the population.

**Question:** how many ENEM candidates in Rio Grande do Sul still take the exam without a computer at home, and
does that weigh on their scores and attendance? Each hypothesis compares an RS parameter with the value for all
of Brazil (H0), using only the sample, and then checks the conclusion against the full population.

## Dataset

- **Source:** INEP (Ministry of Education), [ENEM 2023 microdata](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem),
  [direct download](https://download.inep.gov.br/microdados/microdados_enem_2023.zip) (550 MB zip).
- **Size:** 3,933,955 candidates and 76 variables (Brazil).
- **Population:** the 159,919 candidates whose exam took place in RS (`SG_UF_PROVA = RS`).
- **Sampling unit:** the candidate.
- **Sample:** simple random sample without replacement, n = 384 (95% confidence, 5% margin, p = 0.5,
  finite-population correction), seed 2026.

## Hypotheses (α = 0.05, 95% confidence intervals)

| | Parameter (RS) | H0 (Brazil value) | H1 |
|---|---|---|---|
| P1 | share of candidates with no computer at home (Q024) | π = 0.519 | π < 0.519 |
| P2 | share absent from the math test | π = 0.315 | π ≠ 0.315 |
| M1 | mean math score of candidates with no computer at home | μ = 533.8 | μ < 533.8 |
| M2 | mean essay score | μ = 617.8 | μ > 617.8 |

Proportions use the one-sample z test and the Wald interval; means use the one-sample t test and the t interval,
as in the course notebooks. Panel 3 also repeats the whole procedure on 1,000 new samples to show the power
of each test and the coverage of the intervals.

## The dashboard

`app.py` is a Streamlit app with one page per deliverable:

| Page | URL path | Content |
|---|---|---|
| Base de dados e amostra | `/` | dataset, variables, sample size calculation and sampling procedure |
| Painel 1: população | `/populacao` | descriptive analysis of the population |
| Painel 2: amostra | `/amostra` | the same analysis on the sample, plus sample × population |
| Painel 3: inferência | `/inferencia` | CIs, tests, comparison with the true values, simulation, limitations |

Code is in English; the text on screen is in Portuguese, the course language.

## Files

- `app.py`: navigation between the pages.
- `panels/`: one file per page.
- `common.py`: data loading, sampling, descriptive statistics, tests and charts. Run `uv run python common.py`
  to check the test formulas against statsmodels/scipy.
- `scripts/prepare_data.py`: extracts the RS population and the Brazil reference values from the INEP zip.
- `data/population_rs.parquet`, `data/brazil_reference.csv`: the outputs of that script, committed so the app
  runs without the 550 MB download.

## How to run

```bash
uv sync
uv run streamlit run app.py
```

Or with pip: `pip install -r requirements.txt && streamlit run app.py`.

To rebuild `data/` from the original microdata:

```bash
uv run python scripts/prepare_data.py PATH/microdados_enem_2023.zip
```
