"""Extract the population (ENEM 2023 candidates who took the exam in RS) from INEP's microdata.

Usage: uv run python scripts/prepare_data.py PATH/microdados_enem_2023.zip

Outputs:
- data/population_rs.parquet: candidates with SG_UF_PROVA == 'RS', only the columns used.
- data/brazil_reference.csv: the same indicators for all of Brazil (source of the H0 values).
"""
import sys
import zipfile
from pathlib import Path

import pandas as pd

CSV = "microdados_enem_2023/DADOS/MICRODADOS_ENEM_2023.csv"
COLS = [
    "NU_INSCRICAO", "TP_FAIXA_ETARIA", "TP_SEXO", "TP_COR_RACA", "TP_ST_CONCLUSAO",
    "TP_ESCOLA", "IN_TREINEIRO", "NO_MUNICIPIO_PROVA", "SG_UF_PROVA",
    "TP_PRESENCA_CN", "TP_PRESENCA_CH", "TP_PRESENCA_LC", "TP_PRESENCA_MT",
    "NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO",
    "TP_STATUS_REDACAO", "Q005", "Q006", "Q024", "Q025",
]
OUT = Path(__file__).resolve().parent.parent / "data"


def indicators(df):
    """Per-chunk sums, so Brazil is aggregated without loading 3.9 million rows."""
    present = df["TP_PRESENCA_MT"] == 1
    return pd.Series({
        "candidates": len(df),
        "absent_mt": (df["TP_PRESENCA_MT"] == 0).sum(),
        "no_internet": (df["Q025"] == "A").sum(),
        "no_computer": (df["Q024"] == "A").sum(),
        "n_score_mt": df.loc[present, "NU_NOTA_MT"].notna().sum(),
        "sum_score_mt": df.loc[present, "NU_NOTA_MT"].sum(),
        "n_essay": df["NU_NOTA_REDACAO"].notna().sum(),
        "sum_essay": df["NU_NOTA_REDACAO"].sum(),
    })


def main(zip_path):
    OUT.mkdir(exist_ok=True)
    rs, br = [], []
    with zipfile.ZipFile(zip_path) as z, z.open(CSV) as f:
        for chunk in pd.read_csv(f, sep=";", encoding="latin-1", usecols=COLS, chunksize=500_000):
            br.append(indicators(chunk))
            rs.append(chunk[chunk["SG_UF_PROVA"] == "RS"])
            print(f"{sum(s['candidates'] for s in br):,.0f} rows read", flush=True)
    pop = pd.concat(rs, ignore_index=True)
    pop.to_parquet(OUT / "population_rs.parquet", index=False)

    t = sum(br)
    ref = pd.Series({
        "candidates": t["candidates"],
        "prop_absent_mt": t["absent_mt"] / t["candidates"],
        "prop_no_internet": t["no_internet"] / t["candidates"],
        "prop_no_computer": t["no_computer"] / t["candidates"],
        "mean_score_mt": t["sum_score_mt"] / t["n_score_mt"],
        "mean_essay": t["sum_essay"] / t["n_essay"],
    }, name="brazil")
    ref.to_csv(OUT / "brazil_reference.csv")
    print(f"RS: {len(pop):,} candidates\n{ref}")


if __name__ == "__main__":
    main(sys.argv[1])
