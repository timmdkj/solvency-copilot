"""Génération de données SYNTHÉTIQUES (aucune donnée réelle).

Deux clôtures mensuelles (M-1 = 30/09/2026, M = 31/10/2026) + des séries de fin de mois.
Des événements connus sont injectés pour pouvoir vérifier que le moteur les retrouve (vérité terrain).
Modèle volontairement simple : RWA = exposition x pondération (selon la notation) + ajustement manuel.
Les ordres de grandeur (Tier 1 / RWA, RWA / mesure d'exposition, levier) sont calés « plus ou moins »
sur ceux d'une grande banque de financement et d'investissement, sans prétendre la représenter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

D0 = "2026-09-30"  # clôture M-1
D1 = "2026-10-31"  # clôture M

RATINGS = ["AA", "A", "BBB", "BB", "B", "CCC"]
RATING_P = [0.15, 0.33, 0.32, 0.12, 0.06, 0.02]
# Pondérations ILLUSTRATIVES, du type de celles qu'un modèle interne donnerait par notation
# (pas une table réglementaire).
RW_TABLE = {"AA": 0.15, "A": 0.30, "BBB": 0.55, "BB": 0.90, "B": 1.40, "CCC": 2.20}

SECTORS = ["Énergie", "Immobilier", "Industrie", "Services financiers", "Distribution",
           "Transport", "Santé", "Technologie", "Utilities", "Souverain"]
ENTITIES = ["E1", "E2", "E3", "E4"]  # codes internes de génération (E4 = bloc acquis), jamais affichés
EXISTING, ACQUIRED = "Périmètre existant", "Périmètre acquis"  # seuls libellés visibles

# Périmètre de consolidation : un bloc de positions (« périmètre acquis », ~5 % des contreparties) entre à la clôture M.
PERIMETER = {
    D0: {"E1": True, "E2": True, "E3": True, "E4": False},
    D1: {"E1": True, "E2": True, "E3": True, "E4": True},
}
SPECIAL = {"CP0555", "CP0777", "CP0300"}

# Ratio de levier relié aux contreparties (valeurs FICTIVES, ordres de grandeur illustratifs) :
EXPOSURE_FACTOR = 2.1       # mesure d'exposition = 2,1 x EAD crédit (hors-bilan, dérivés, financements...)
TIER1_SHARE_OF_RWA = 0.15   # Tier 1 = 15 % du RWA de M-1 (constant entre les deux clôtures)

SPIKE_DATE = "2026-03-31"   # pic de fin de trimestre injecté sur la mesure d'exposition


def make_bundle(seed: int = 42, n: int = 1200) -> dict:
    rng = np.random.default_rng(seed)
    ids = [f"CP{i:04d}" for i in range(1, n + 1)]
    entity = rng.choice(ENTITIES, size=n, p=[0.54, 0.22, 0.19, 0.05])
    rating = [RATINGS[i] for i in rng.choice(len(RATINGS), size=n, p=RATING_P)]
    sector = rng.choice(SECTORS, size=n)
    ead = np.clip(rng.lognormal(np.log(35), 0.9, n), 1, 700)
    ead = np.where(entity == "E4", ead * 0.8, ead)

    d0 = pd.DataFrame({"cpty_id": ids, "entity": entity, "sector": sector, "rating": rating, "ead": ead})
    sov = d0["sector"] == "Souverain"
    d0.loc[sov, "rating"] = rng.choice(["AA", "A"], size=int(sov.sum()))

    # Positions « spéciales » (événements injectés), toutes à E1
    overrides = {
        "CP0555": dict(sector="Industrie", rating="BBB", ead=25.0),
        "CP0777": dict(sector="Souverain", rating="AA", ead=20.0),
        "CP0300": dict(sector="Distribution", rating="A", ead=40.0),
    }
    for cid, vals in overrides.items():
        m = d0["cpty_id"] == cid
        d0.loc[m, "entity"] = "E1"
        for k, v in vals.items():
            d0.loc[m, k] = v

    d0["overlay"] = 0.0  # ajustement manuel de RWA (M€)
    d0["date"] = D0
    d0["in_scope"] = d0["entity"].map(PERIMETER[D0]).astype(bool)

    # ------------------------- M : application des événements -------------------------
    d1 = d0.copy()
    d1["date"] = D1
    d1["in_scope"] = d1["entity"].map(PERIMETER[D1]).astype(bool)

    def shift(cid: str, step: int):
        m = d1["cpty_id"] == cid
        d1.loc[m, "rating"] = RATINGS[RATINGS.index(d1.loc[m, "rating"].iloc[0]) + step]

    # 1) Dégradations (4) et relèvements (3) de notation d'un cran, sur des contreparties de E1
    base = d1[(d1["entity"] == "E1") & (~d1["cpty_id"].isin(SPECIAL)) & (d1["sector"] != "Souverain")]
    down_pool = base[base["rating"].isin(["AA", "A", "BBB", "BB"])].sort_values("ead", ascending=False)
    downgraded = list(down_pool["cpty_id"].head(4))
    for cid in downgraded:
        shift(cid, +1)
    up_pool = base[(base["rating"].isin(["A", "BBB", "BB", "B"])) & (~base["cpty_id"].isin(downgraded))]
    upgraded = list(up_pool.sort_values("ead", ascending=False)["cpty_id"].iloc[10:13])
    for cid in upgraded:
        shift(cid, -1)

    # 2) Variations d'encours sur toutes les positions (utilisations, remboursements partiels), ~3 %
    drift = np.exp(rng.normal(0, 0.03, len(d1)))
    d1["ead"] = np.where(d1["cpty_id"].isin(SPECIAL), d1["ead"], d1["ead"] * drift)

    # 3) Anomalies injectées
    d1.loc[d1["cpty_id"] == "CP0555", "ead"] = 100.0               # EAD x4, notation inchangée
    d1.loc[d1["cpty_id"] == "CP0777", "rating"] = "CCC"            # souverain passé à CCC (donnée incohérente)
    d1.loc[d1["cpty_id"] == "CP0300", "overlay"] = 40.0           # ajustement manuel de 40 M€ de RWA

    # 4) Sorties de positions (échéances, remboursements)
    pool = d1[(d1["in_scope"]) & (d1["entity"] != "E4") & (~d1["cpty_id"].isin(SPECIAL))
              & (~d1["cpty_id"].isin(downgraded + upgraded))]
    exits = list(rng.choice(pool["cpty_id"].to_numpy(), size=40, replace=False))
    d1 = d1[~d1["cpty_id"].isin(exits)].copy()

    # 5) Nouvelles opérations (dont une grosse, en Énergie)
    rows = []
    for k in range(2, 42):
        rows.append(dict(
            cpty_id=f"CP{9000 + k:04d}", entity=str(rng.choice(["E1", "E2", "E3"], p=[0.55, 0.25, 0.20])),
            sector=str(rng.choice(SECTORS[:-1])), rating=RATINGS[int(rng.choice(len(RATINGS), p=RATING_P))],
            ead=float(np.clip(rng.lognormal(np.log(30), 0.7), 2, 250)), overlay=0.0))
    rows.append(dict(cpty_id="CP9001", entity="E1", sector="Énergie", rating="BBB", ead=600.0, overlay=0.0))
    new = pd.DataFrame(rows)
    new["date"] = D1
    new["in_scope"] = new["entity"].map(PERIMETER[D1]).astype(bool)
    d1 = pd.concat([d1, new[d1.columns]], ignore_index=True)

    for df in (d0, d1):  # libellés neutres : aucune ville n'apparaît dans les données
        df["entity"] = np.where(df["entity"] == "E4", ACQUIRED, EXISTING)
    series = _make_series(rng, d0, d1)
    truth = {
        "new_large": "CP9001", "downgraded": downgraded, "upgraded": upgraded, "exits": exits,
        "perimeter_entity": ACQUIRED, "anomaly_ead": "CP0555", "anomaly_rating": "CP0777",
        "overlay_id": "CP0300", "overlay_amount": 40.0, "series_spike_date": SPIKE_DATE,
    }
    return {"s0": d0.reset_index(drop=True), "s1": d1.reset_index(drop=True), "series": series, "truth": truth}


def _make_series(rng: np.random.Generator, d0: pd.DataFrame, d1: pd.DataFrame) -> pd.DataFrame:
    """Séries de fin de mois (24 clôtures). Les deux derniers points sont calculés à partir des
    snapshots : RWA = somme des RWA ; mesure d'exposition = EXPOSURE_FACTOR x EAD en périmètre."""
    from .engine import total_rwa

    dates = pd.date_range(end=D1, periods=24, freq=pd.offsets.MonthEnd())
    n = len(dates)
    rwa = np.zeros(n)
    rwa[-1], rwa[-2] = total_rwa(d1), total_rwa(d0)
    r = rng.normal(0.004, 0.02, n - 2)
    for i in range(n - 3, -1, -1):
        rwa[i] = rwa[i + 1] / (1 + r[i])

    ead0 = d0.loc[d0["in_scope"], "ead"].sum()
    ead1 = d1.loc[d1["in_scope"], "ead"].sum()
    expo = np.zeros(n)  # mesure d'exposition (Md€)
    expo[-1], expo[-2] = EXPOSURE_FACTOR * ead1 / 1000, EXPOSURE_FACTOR * ead0 / 1000
    rr = rng.normal(0.004, 0.010, n - 2)
    for i in range(n - 3, -1, -1):
        expo[i] = expo[i + 1] / (1 + rr[i])
    k = list(dates).index(pd.Timestamp(SPIKE_DATE))
    expo[k] *= 1.08  # pic de fin de trimestre (un seul mois)
    rwa[k] *= 1.06   # le même effet de fin de trimestre se retrouve, atténué, sur le RWA

    t1 = TIER1_SHARE_OF_RWA * rwa[-2] / 1000
    tier1 = t1 * (1 + rng.normal(0, 0.004, n))
    tier1[-2:] = t1  # Tier 1 inchangé entre M-1 et M : la variation vient de l'exposition
    out = pd.DataFrame({"date": dates, "rwa_total": rwa, "tier1": tier1, "exposure": expo})
    out["leverage"] = out["tier1"] / out["exposure"] * 100
    return out
