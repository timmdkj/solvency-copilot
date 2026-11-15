"""Petit glossaire pédagogique du prototype (pas un texte réglementaire)."""
from __future__ import annotations

import re
import unicodedata

NOTE = "Définition pédagogique propre au prototype, pas un texte réglementaire."

# terme : (alias, définition)
GLOSSARY = {
    "Effet de périmètre": (
        ["effet de perimetre", "perimetre"],
        "Variation de RWA due à l'entrée ou à la sortie d'un bloc de positions (entité, portefeuille acquis ou cédé) dans le périmètre de consolidation. "
        "On l'isole pour lire les autres effets à périmètre constant."),
    "Nouvelles opérations": (
        ["nouvelles operations", "nouvelle operation", "nouvelles positions"],
        "RWA des positions apparues entre M-1 et M, hors entrée de périmètre."),
    "Sorties de positions": (
        ["sorties de positions", "sorties", "sortie"],
        "RWA des positions présentes à M-1 et disparues à M (échéance, remboursement, cession)."),
    "Variation d'encours": (
        ["variation d encours", "encours", "effet volume", "volume"],
        "Variation de l'exposition d'une position présente aux deux dates, valorisée avec la pondération de M-1."),
    "Changement de notation": (
        ["changement de notation", "notation", "migration", "degradation"],
        "Variation de RWA due à un changement de notation, donc de pondération, sur une position présente "
        "aux deux dates (valorisée sur l'exposition de J)."),
    "Résiduel": (
        ["residuel", "ajustement manuel", "ajustements manuels"],
        "Part de la variation non expliquée par les autres effets. Ici, ce sont des ajustements manuels. "
        "Un résiduel élevé signale une analyse incomplète."),
    "Réconciliation": (
        ["reconciliation", "reconcilier"],
        "Contrôle que la somme des effets est égale à la variation totale de RWA. "
        "En cas d'écart, la décomposition est incorrecte."),
    "RWA": (
        ["rwa", "actifs ponderes", "risk weighted assets"],
        "Risk-Weighted Assets (actifs pondérés des risques) : exposition multipliée par une pondération. "
        "Dans ce prototype, calcul illustratif et simplifié, pas un calcul réglementaire."),
    "EAD": (
        ["ead", "exposure at default", "exposition"],
        "Exposure At Default : montant de l'exposition en cas de défaut, exprimé ici en M€."),
    "Pondération": (
        ["ponderation", "risk weight", "poids de risque"],
        "Pourcentage appliqué à l'exposition pour obtenir le RWA. Ici, une table simple selon la notation "
        "(de 20 % pour AA à 200 % pour CCC), purement illustrative."),
    "Ratio de levier": (
        ["ratio de levier", "levier", "leverage"],
        "Tier 1 divisé par la mesure d'exposition totale, sans pondération par les risques. "
        "Il complète les ratios fondés sur les RWA."),
    "Tier 1": (
        ["tier 1", "tier1"],
        "Fonds propres de catégorie 1 : capital de meilleure qualité, numérateur du ratio de levier."),
    "Ratio CET1": (
        ["ratio cet1", "cet1"],
        "Fonds propres CET1 divisés par les RWA. À fonds propres constants, une hausse de RWA fait baisser le "
        "ratio. Le prototype ne calcule pas ce ratio."),
    "Anomalie": (
        ["anomalie", "anomalies", "isolation forest", "atypique"],
        "Position dont la variation depuis la clôture précédente est inhabituelle (Isolation Forest et z-score robuste). "
        "C'est un signal à vérifier, pas une erreur confirmée."),
    "Clôtures M et M-1": (
        ["m-1", "m 1", "cloture", "cloture mensuelle", "mensuel", "mensuelle"],
        "M est la clôture mensuelle de référence, M-1 la précédente. Ici : M-1 = 30/09/2026 et M = 31/10/2026 "
        "(données synthétiques). Les publications officielles, elles, sont trimestrielles."),
    "Pilier 3 (CR8)": (
        ["pilier 3", "cr8", "pillar 3"],
        "Publication réglementaire des banques. Le tableau CR8 explique les variations de RWA du risque de "
        "crédit en approche interne entre deux dates de publication, par catégories : taille des actifs, qualité "
        "des actifs, mises à jour de modèles, méthodologie et politique, acquisitions et cessions, change, autres. "
        "Le prototype regroupe ses effets selon ces catégories (liste citée de mémoire)."),
}


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def define(term: str) -> dict:
    """Cherche le terme le plus proche (alias le plus long contenu dans la requête)."""
    q = " " + _norm(term) + " "
    best, best_len = None, 0
    for name, (aliases, _) in GLOSSARY.items():
        for a in [name] + aliases:
            na = _norm(a)
            if f" {na} " in q and len(na) > best_len:
                best, best_len = name, len(na)
    if best is None:
        return {"found": False, "available": list(GLOSSARY)}
    return {"found": True, "term": best, "definition": GLOSSARY[best][1], "note": NOTE}
