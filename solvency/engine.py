"""Moteur de calcul : RWA illustratif, explication des variations, anomalies, requêtes.

Tout est déterministe. Le modèle de langage (llm.py) ne calcule rien : il appelle ces fonctions.
Modèle volontairement simple : RWA = exposition x pondération (selon la notation) + ajustement manuel.
C'est ILLUSTRATIF, pas un calcul réglementaire.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .data import D0, D1, EXPOSURE_FACTOR, RW_TABLE

HISTORY_PATH = Path(__file__).resolve().parent.parent / "data" / "history.json"


def fr(x: float, nd: int = 0, sign: bool = False) -> str:
    """Format français : espace fine pour les milliers, virgule décimale."""
    txt = f"{x:{'+' if sign else ''},.{nd}f}"
    return txt.replace(",", " ").replace(".", ",")


def fdate(iso: str) -> str:
    """2026-10-31 -> 31/10/2026."""
    y, m, d = iso.split("-")
    return f"{d}/{m}/{y}"


# Correspondance avec les catégories habituelles des tableaux de flux du Pilier 3 (type CR8).
PILLAR3 = {"perimeter": "Acquisitions et cessions", "new": "Taille des actifs", "exit": "Taille des actifs",
           "volume": "Taille des actifs", "rating": "Qualité des actifs", "residual": "Autres"}
PILLAR3_ORDER = ["Taille des actifs", "Qualité des actifs", "Acquisitions et cessions", "Autres"]


def pillar3_view(result: dict) -> dict:
    """Regroupe les effets selon les catégories d'un tableau de flux de type Pilier 3 (CR8)."""
    tot = {k: 0.0 for k in PILLAR3_ORDER}
    for e in result["effects"]:
        tot[PILLAR3[e["key"]]] += e["value"]
    return {"start": result["total0"], "end": result["total1"],
            "rows": [{"category": k, "value": tot[k]} for k in PILLAR3_ORDER]}


EFFECTS = [
    ("perimeter", "Effet de périmètre"),
    ("new", "Nouvelles opérations"),
    ("exit", "Sorties de positions"),
    ("volume", "Variation d'encours"),
    ("rating", "Changement de notation"),
    ("residual", "Résiduel / ajustements manuels"),
]
LABELS = dict(EFFECTS)


# --------------------------------------------------------------------------- RWA illustratif
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la pondération (selon la notation) et le RWA déclaré (exposition x pondération + ajustement)."""
    out = df.copy()
    out["rw"] = out["rating"].map(RW_TABLE).astype(float)
    out["rwa"] = np.where(out["in_scope"].astype(bool), out["ead"] * out["rw"] + out["overlay"], 0.0)
    return out


def total_rwa(df: pd.DataFrame) -> float:
    return float(enrich(df)["rwa"].sum())


# --------------------------------------------------------------------------- Explication des variations
def explain_variation(s0: pd.DataFrame, s1: pd.DataFrame, top_n: int = 5) -> dict:
    """Décomposition de la variation de RWA entre deux snapshots.

    1. Effet de périmètre : positions d'une entité qui entre ou sort du périmètre (isolé en premier).
    2. Nouvelles opérations / sorties : positions absentes d'une des deux dates.
    3. Pour les positions présentes aux deux dates et dans le périmètre :
         RWA1 - RWA0 = variation d'encours  (EAD1 - EAD0) x pondération0
                     + changement de notation EAD1 x (pondération1 - pondération0)
                     + ajustement manuel (résiduel).
    L'ordre (encours puis notation) est choisi, documenté et fixe : il détermine où va le terme croisé.
    """
    a, b = enrich(s0), enrich(s1)
    cols = ["cpty_id", "entity", "sector", "rating", "ead", "rw", "overlay", "rwa"]
    m = a[cols].merge(b[cols], on="cpty_id", how="outer", suffixes=("0", "1"), indicator=True)

    scope0_by_ent = s0.groupby("entity")["in_scope"].first().astype(bool).to_dict()
    scope1_by_ent = s1.groupby("entity")["in_scope"].first().astype(bool).to_dict()
    m["entity"] = m["entity0"].fillna(m["entity1"])
    m["sector"] = m["sector0"].fillna(m["sector1"])
    scope0 = m["entity"].map(scope0_by_ent).astype(bool)
    scope1 = m["entity"].map(scope1_by_ent).astype(bool)
    present0 = m["_merge"] != "right_only"
    present1 = m["_merge"] != "left_only"
    rwa0 = m["rwa0"].fillna(0.0)
    rwa1 = m["rwa1"].fillna(0.0)

    changed = scope0 != scope1
    cont = (~changed) & present0 & present1 & scope1
    eff = pd.DataFrame(0.0, index=m.index, columns=[k for k, _ in EFFECTS])

    eff.loc[changed, "perimeter"] = (rwa1 - rwa0)[changed]
    new_mask = (~changed) & (~present0) & present1
    eff.loc[new_mask, "new"] = rwa1[new_mask]
    exit_mask = (~changed) & present0 & (~present1)
    eff.loc[exit_mask, "exit"] = -rwa0[exit_mask]

    c = m[cont]
    eff.loc[cont, "volume"] = ((c["ead1"] - c["ead0"]) * c["rw0"]).to_numpy()
    eff.loc[cont, "rating"] = (c["ead1"] * (c["rw1"] - c["rw0"])).to_numpy()
    eff.loc[cont, "residual"] = (c["overlay1"] - c["overlay0"]).to_numpy()

    total0, total1 = float(rwa0.sum()), float(rwa1.sum())
    delta = total1 - total0
    effects = [{"key": k, "label": LABELS[k], "pillar3": PILLAR3[k], "value": float(eff[k].sum())} for k, _ in EFFECTS]
    gap = abs(delta - sum(e["value"] for e in effects))

    per_id_total = rwa1 - rwa0
    driver = eff.abs().idxmax(axis=1)
    top = per_id_total.abs().sort_values(ascending=False).head(top_n).index
    contributors = []
    for i in top:
        row = m.loc[i]
        contributors.append({
            "cpty_id": row["cpty_id"], "sector": row["sector"], "entity": row["entity"],
            "delta": float(per_id_total.loc[i]), "driver": LABELS[driver[i]],
            "explanation": _explain_row(row, driver[i]),
        })

    return {
        "date0": s0["date"].iloc[0], "date1": s1["date"].iloc[0],
        "total0": total0, "total1": total1, "delta": delta,
        "delta_pct": delta / total0 * 100 if total0 else 0.0,
        "effects": effects, "reconciliation_gap": float(gap), "reconciled": bool(gap < 1e-6),
        "residual": next(e["value"] for e in effects if e["key"] == "residual"),
        "top_contributors": contributors,
    }


def _explain_row(row: pd.Series, driver: str) -> str:
    if driver == "perimeter":
        return "Entrée dans le périmètre (portefeuille acquis)"
    if driver == "new":
        return f"Nouvelle opération ({row['sector']}, notation {row['rating1']}, EAD {fr(row['ead1'])} M€)"
    if driver == "exit":
        return "Position sortie (échéance ou remboursement)"
    if driver == "rating":
        return (f"Changement de notation {row['rating0']} → {row['rating1']} "
                f"(pondération {fr(row['rw0'] * 100)} % → {fr(row['rw1'] * 100)} %)")
    if driver == "volume":
        return f"Variation d'EAD de {fr((row['ead1'] / row['ead0'] - 1) * 100, 0, True)} %"
    if driver == "residual":
        return "Ajustement manuel"
    return LABELS.get(driver, driver)


# --------------------------------------------------------------------------- Anomalies
_FEATURES = ["d_ead", "d_rw_rel"]  # variations inhabituelles depuis la clôture précédente (un changement sectoriel s'annule)
_SCALE_FLOOR = {"d_ead": 0.02, "d_rw_rel": 0.05}


def detect_anomalies(s0: pd.DataFrame, s1: pd.DataFrame, top_n: int = 8,
                     contamination: float = 0.01, seed: int = 0) -> dict:
    """Isolation Forest sur les variations de chaque contrepartie + raison lisible (z-score robuste)."""
    a, b = enrich(s0), enrich(s1)
    b = b[b["in_scope"]].copy()
    prev = a.set_index("cpty_id")
    b["ead_prev"] = b["cpty_id"].map(prev["ead"])
    b["rw_prev"] = b["cpty_id"].map(prev["rw"])
    b["rating_prev"] = b["cpty_id"].map(prev["rating"])
    b["d_ead"] = np.log(b["ead"] / b["ead_prev"]).fillna(0.0)
    d_rw = np.log(b["rw"] / b["rw_prev"])  # NaN pour les positions nouvelles (pas de clôture précédente comparable)
    b["d_rw_rel"] = (d_rw - d_rw.groupby(b["sector"]).transform("median")).fillna(0.0)

    X = b[_FEATURES].to_numpy()
    # max_samples=1.0 : les variations non nulles sont rares, il faut que chaque arbre les voie
    model = IsolationForest(n_estimators=300, max_samples=1.0, contamination=contamination,
                            random_state=seed).fit(X)
    b["score"] = -model.score_samples(X)
    n_flag = max(1, int(round(contamination * len(b))))
    b = b.sort_values("score", ascending=False).reset_index(drop=True)

    med = b[_FEATURES].median()
    scale = {f: max(1.4826 * (b[f] - med[f]).abs().median(), _SCALE_FLOOR[f]) for f in _FEATURES}
    items = []
    for i, row in b.head(top_n).iterrows():
        z = {f: (row[f] - med[f]) / scale[f] for f in _FEATURES}
        top_f = sorted(_FEATURES, key=lambda f: abs(z[f]), reverse=True)[:2]
        reasons = [_reason(f, row, z[f]) for f in top_f if abs(z[f]) >= 3]
        if reasons and row["sector"] == "Souverain" and row["rw"] >= 1.0:
            reasons.append("notation inhabituelle pour un souverain")
        items.append({
            "cpty_id": row["cpty_id"], "sector": row["sector"], "entity": row["entity"],
            "rating": row["rating"], "ead": float(row["ead"]),
            "score": float(row["score"]), "flagged": bool(i < n_flag),
            "reason": " ; ".join(reasons) if reasons else "Profil globalement atypique",
        })
    return {"method": "Isolation Forest + z-score robuste", "n_scored": int(len(b)),
            "contamination": contamination, "items": items}


def _reason(f: str, row: pd.Series, z: float) -> str:
    if f == "d_ead":
        return (f"EAD multipliée par {fr(np.exp(row['d_ead']), 1)} en un mois" if z > 0
                else "forte baisse d'EAD en un mois")
    return (f"pondération passée de {fr(row['rw_prev'] * 100)} % à {fr(row['rw'] * 100)} % "
            f"(notation {row['rating_prev']} → {row['rating']})" if z > 0
            else "forte baisse de pondération en un mois")


def detect_series_anomalies(series: pd.DataFrame, col: str, threshold: float = 3.5) -> list[dict]:
    """Variations mensuelles aberrantes (z-score robuste, MAD)."""
    x = series.set_index("date")[col]
    d = x.pct_change().dropna()
    med = d.median()
    mad = max(1.4826 * (d - med).abs().median(), 1e-9)
    z = (d - med) / mad
    out = []
    for dt, zz in z[z.abs() > threshold].items():
        out.append({"date": dt.strftime("%Y-%m-%d"), "value": float(x.loc[dt]),
                    "change_pct": float(d.loc[dt] * 100), "z": float(zz)})
    return out


def rwa_series_summary(series: pd.DataFrame) -> dict:
    """Variations mensuelles aberrantes du RWA total, avec analyse déjà réalisée le cas échéant."""
    anomalies = detect_series_anomalies(series, "rwa_total")
    for x in anomalies:
        past = find_analyses("rwa_credit", x["date"], window_days=5)
        x["analyse"] = past[0]["title"] if past else "aucune"
    d = series["rwa_total"].pct_change().iloc[-1] * 100
    return {"anomalies": anomalies, "change_pct": float(d), "z_last": last_change_z(series, "rwa_total")}


def last_change_z(series: pd.DataFrame, col: str) -> float:
    """Écart robuste (en « écarts-types », MAD) de la dernière variation mensuelle."""
    d = series.set_index("date")[col].pct_change().dropna()
    mad = max(1.4826 * (d - d.median()).abs().median(), 1e-9)
    return float((d.iloc[-1] - d.median()) / mad)


def leverage_summary(series: pd.DataFrame) -> dict:
    last, prev = series.iloc[-1], series.iloc[-2]
    anomalies = detect_series_anomalies(series, "leverage")
    for x in anomalies:
        past = find_analyses("leverage_ratio", x["date"], window_days=5)
        x["analyse"] = past[0]["title"] if past else "aucune"
    return {
        "date": last["date"].strftime("%Y-%m-%d"),
        "leverage_pct": float(last["leverage"]), "previous_pct": float(prev["leverage"]),
        "change_bp": float((last["leverage"] - prev["leverage"]) * 100),
        "tier1_bn": float(last["tier1"]), "exposure_bn": float(last["exposure"]),
        "exposure_prev_bn": float(prev["exposure"]),
        "change_pct": float((last["leverage"] / prev["leverage"] - 1) * 100),
        "z_last": last_change_z(series, "leverage"),
        "anomalies": anomalies,
    }


LEV_EFFECTS = [
    ("perimeter", "Effet de périmètre"),
    ("new", "Nouvelles opérations"),
    ("exit", "Sorties de positions"),
    ("volume", "Variation d'encours"),
]


def explain_leverage(s0: pd.DataFrame, s1: pd.DataFrame, series: pd.DataFrame, top_n: int = 3) -> dict:
    """Origine de la variation du ratio de levier entre M-1 et M, par contrepartie et par périmètre.

    Le Tier 1 est identique aux deux dates : toute la variation vient de la mesure d'exposition
    (EXPOSURE_FACTOR x EAD en périmètre). La variation en points de base est répartie entre les effets
    au prorata de leur contribution à la variation d'exposition.
    """
    a, b = enrich(s0), enrich(s1)
    cols = ["cpty_id", "entity", "sector", "rating", "ead"]
    m = a[cols].merge(b[cols], on="cpty_id", how="outer", suffixes=("0", "1"), indicator=True)
    scope0_by_ent = s0.groupby("entity")["in_scope"].first().astype(bool).to_dict()
    scope1_by_ent = s1.groupby("entity")["in_scope"].first().astype(bool).to_dict()
    m["entity"] = m["entity0"].fillna(m["entity1"])
    m["sector"] = m["sector0"].fillna(m["sector1"])
    scope0 = m["entity"].map(scope0_by_ent).astype(bool)
    scope1 = m["entity"].map(scope1_by_ent).astype(bool)
    present0, present1 = m["_merge"] != "right_only", m["_merge"] != "left_only"
    e0 = m["ead0"].fillna(0.0).where(scope0, 0.0)
    e1 = m["ead1"].fillna(0.0).where(scope1, 0.0)

    changed = scope0 != scope1
    cont = (~changed) & present0 & present1 & scope1
    eff = pd.DataFrame(0.0, index=m.index, columns=[k for k, _ in LEV_EFFECTS])
    eff.loc[changed, "perimeter"] = (e1 - e0)[changed]
    nm = (~changed) & (~present0) & present1
    eff.loc[nm, "new"] = e1[nm]
    xm = (~changed) & present0 & (~present1)
    eff.loc[xm, "exit"] = -e0[xm]
    eff.loc[cont, "volume"] = (e1 - e0)[cont]

    lev0, lev1 = float(series["leverage"].iloc[-2]), float(series["leverage"].iloc[-1])
    delta_bp = (lev1 - lev0) * 100
    d_ead = float((e1 - e0).sum())                 # M€ d'EAD
    d_expo = EXPOSURE_FACTOR * d_ead / 1000        # Md€ de mesure d'exposition
    share = (lambda x: x / d_ead) if d_ead else (lambda x: 0.0)
    effects = [{"key": k, "label": lab, "exposure_bn": EXPOSURE_FACTOR * float(eff[k].sum()) / 1000,
                "value": delta_bp * share(float(eff[k].sum()))} for k, lab in LEV_EFFECTS]

    per_id = e1 - e0
    ent = per_id.groupby(m["entity"]).sum()
    by_entity = [{"entity": k, "exposure_bn": EXPOSURE_FACTOR * float(v) / 1000, "value": delta_bp * share(float(v))}
                 for k, v in ent.reindex(ent.abs().sort_values(ascending=False).index).items()]
    driver = eff.abs().idxmax(axis=1)
    contributors = []
    for i in per_id.abs().sort_values(ascending=False).head(top_n).index:
        row = m.loc[i]
        contributors.append({
            "cpty_id": row["cpty_id"], "entity": row["entity"], "sector": row["sector"],
            "delta_ead": float(per_id.loc[i]), "contribution_bp": delta_bp * share(float(per_id.loc[i])),
            "explanation": _explain_lev_row(row, driver[i]),
        })
    gap = abs(delta_bp - sum(e["value"] for e in effects))
    return {
        "date0": s0["date"].iloc[0], "date1": s1["date"].iloc[0], "lev0": lev0, "lev1": lev1,
        "delta": delta_bp, "delta_exposure_bn": d_expo, "tier1_bn": float(series["tier1"].iloc[-1]),
        "exposure0_bn": float(series["exposure"].iloc[-2]), "exposure1_bn": float(series["exposure"].iloc[-1]),
        "effects": effects, "by_entity": by_entity, "contributors": contributors,
        "reconciliation_gap": float(gap), "reconciled": bool(gap < 1e-6),
    }


def _explain_lev_row(row: pd.Series, driver: str) -> str:
    if driver == "perimeter":
        return "Entrée dans le périmètre (portefeuille acquis)"
    if driver == "new":
        return f"Nouvelle opération ({row['sector']}, EAD {fr(row['ead1'])} M€)"
    if driver == "exit":
        return "Position sortie (échéance ou remboursement)"
    return f"Variation d'EAD de {fr((row['ead1'] / row['ead0'] - 1) * 100, 0, True)} %"


# --------------------------------------------------------------------------- Requêtes SQL (liste blanche, lecture seule)
QUERIES = {
    "rwa_by_sector": ("RWA par secteur",
                      "SELECT sector AS secteur, ROUND(SUM(rwa), 0) AS rwa_meur FROM exposures "
                      "WHERE date = :d AND in_scope = 1 GROUP BY sector ORDER BY rwa_meur DESC"),
    "rwa_by_rating": ("RWA par notation",
                      "SELECT rating AS notation, ROUND(SUM(rwa), 0) AS rwa_meur FROM exposures "
                      "WHERE date = :d AND in_scope = 1 GROUP BY rating ORDER BY rwa_meur DESC"),
    "rwa_by_entity": ("RWA par périmètre",
                      "SELECT entity AS perimetre, ROUND(SUM(rwa), 0) AS rwa_meur FROM exposures "
                      "WHERE date = :d AND in_scope = 1 GROUP BY entity ORDER BY rwa_meur DESC"),
    "top_counterparties": ("Principales contreparties par RWA",
                           "SELECT cpty_id AS contrepartie, sector AS secteur, rating AS notation, "
                           "ROUND(ead, 0) AS ead_meur, ROUND(rwa, 0) AS rwa_meur FROM exposures "
                           "WHERE date = :d AND in_scope = 1 ORDER BY rwa DESC LIMIT 10"),
    "rwa_total_series": ("RWA total, 24 dernières clôtures mensuelles",
                         "SELECT date, ROUND(rwa_total, 0) AS rwa_meur FROM series "
                         "ORDER BY date DESC LIMIT 24"),
}
ALLOWED_DATES = (D0, D1)


def build_db(bundle: dict) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:", check_same_thread=False)
    ex = pd.concat([enrich(bundle["s0"]), enrich(bundle["s1"])], ignore_index=True)
    ex["in_scope"] = ex["in_scope"].astype(int)
    ex[["date", "cpty_id", "entity", "sector", "rating", "ead", "rw", "rwa", "in_scope"]].to_sql(
        "exposures", con, index=False)
    s = bundle["series"].copy()
    s["date"] = s["date"].dt.strftime("%Y-%m-%d")
    s.to_sql("series", con, index=False)
    con.execute("PRAGMA query_only = ON")  # lecture seule
    return con


def query_metric(con: sqlite3.Connection, name: str, date: str | None = None) -> dict:
    if name not in QUERIES:
        raise ValueError(f"Requête inconnue : {name}. Disponibles : {', '.join(QUERIES)}")
    date = date or D1
    if date not in ALLOWED_DATES:
        raise ValueError(f"Date indisponible : {date}. Dates : {', '.join(ALLOWED_DATES)}")
    title, sql = QUERIES[name]
    cur = con.execute(sql, {"d": date} if ":d" in sql else {})
    cols = [c[0] for c in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return {"name": name, "title": title, "date": date, "sql": sql.replace(":d", f"'{date}'"),
            "columns": cols, "rows": rows}


# --------------------------------------------------------------------------- Historique des analyses
def find_analyses(metric: str, date: str, window_days: int = 5) -> list[dict]:
    """Analyses déjà réalisées sur cette métrique, proches de la date demandée."""
    try:
        entries = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    d = pd.Timestamp(date)
    return [e for e in entries
            if e.get("metric") == metric and abs((pd.Timestamp(e["date"]) - d).days) <= window_days]
