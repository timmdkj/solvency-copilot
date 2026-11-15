"""Couche conversationnelle.

Principe : le modèle de langage choisit un OUTIL et rédige la réponse à partir des chiffres
retournés. Il ne calcule rien. Chaque chiffre cité est ensuite recontrôlé contre les résultats.
Sans clé API (ou en cas d'erreur), un mode démo applique un routage par mots-clés et des
gabarits de texte, avec exactement les mêmes outils et les mêmes chiffres.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass

import requests

from .data import D0, D1
from .glossary import define
from .engine import (PILLAR3_ORDER, QUERIES, detect_anomalies, explain_leverage, explain_variation, fdate, fr,
                     find_analyses, leverage_summary, pillar3_view, query_metric)

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

REFUSAL = (
    "Cette question sort de mon périmètre actuel. Je peux répondre sur la variation du RWA crédit, "
    "les anomalies de contreparties, le ratio de levier, quelques vues (secteur, notation, périmètre, "
    "principales contreparties, série de RWA) et expliquer les termes du prototype (effet de périmètre, "
    "notation, résiduel…). Je préfère ne pas inventer de chiffre."
)

SYSTEM_PROMPT = f"""Tu es l'assistant d'un prototype de pilotage des métriques de solvabilité.
Les données sont SYNTHÉTIQUES et le calcul de RWA est ILLUSTRATIF (pas un calcul réglementaire).
Clôtures mensuelles disponibles : M-1 = {D0}, M = {D1}. Unités : RWA et EAD en M€, levier en %.

Règles strictes :
1. Pour toute question chiffrée, appelle un outil. N'invente aucun chiffre, ne calcule rien toi-même.
2. Réponds en français, en 120 mots maximum, en citant les chiffres EXACTEMENT comme les outils les fournissent.
3. Pour « c'est quoi X ? » sur un terme du prototype, appelle define_term et reprends la définition fournie.
   Si la question sort du périmètre (autres métriques, VaR, FRTB, avis, etc.), n'appelle aucun outil et dis-le simplement.
4. Si une analyse a déjà été réalisée sur la métrique (champ analyses_existantes), mentionne-la. Si la valeur est « aucune », n'en parle PAS : n'écris jamais « aucune analyse » ni une phrase équivalente.
5. Ne présente jamais un résultat comme un calcul réglementaire officiel."""

TOOLS = [
    {"type": "function", "function": {
        "name": "explain_variation",
        "description": "Explique la variation du RWA crédit entre les clôtures M-1 et M : décomposition par effets "
                       "(périmètre, nouvelles opérations, sorties, volume, change, PD, LGD, maturité, résiduel), "
                       "contrôle de réconciliation et principaux contributeurs.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "detect_anomalies",
        "description": "Liste les contreparties atypiques (Isolation Forest) avec la raison de chaque alerte.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "leverage_ratio",
        "description": "Ratio de levier de la clôture M, variation en points de base, origine de la variation "
                       "(périmètre, nouvelles opérations, sorties, encours, principales contreparties), "
                       "Tier 1, mesure d'exposition et variations mensuelles aberrantes sur 24 mois.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "define_term",
        "description": "Définition d'un terme du prototype (effet de périmètre, notation, résiduel, RWA, EAD, "
                       "pondération, ratio de levier, anomalie…). Passe le terme demandé dans 'term'.",
        "parameters": {"type": "object", "properties": {"term": {"type": "string"}}, "required": ["term"]}}},
    {"type": "function", "function": {
        "name": "query_metric",
        "description": "Requête SQL prédéfinie en lecture seule. name parmi : "
                       + ", ".join(f"{k} ({v[0]})" for k, v in QUERIES.items()),
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "enum": list(QUERIES)},
            "date": {"type": "string", "enum": [D0, D1], "description": "clôture M-1 ou M (défaut : M)"}},
            "required": ["name"]}}},
]


@dataclass
class Ctx:
    bundle: dict
    con: object  # connexion SQLite en lecture seule


# --------------------------------------------------------------------------- Outils
def _r(x, nd=0):
    return round(float(x), nd)


def run_tool(name: str, args: dict, ctx: Ctx) -> dict:
    """Exécute un outil. 'facts' = ce que voit le modèle ; 'display' = ce que voit l'interface."""
    b = ctx.bundle
    if name == "explain_variation":
        r = explain_variation(b["s0"], b["s1"])
        past = find_analyses("rwa_credit", D1)
        facts = {
            "metrique": "RWA crédit (illustratif)", "date_precedente": r["date0"], "date": r["date1"],
            "rwa_precedent_meur": _r(r["total0"]), "rwa_meur": _r(r["total1"]),
            "variation_meur": _r(r["delta"]), "variation_pct": _r(r["delta_pct"], 1),
            "effets_meur": {e["label"]: _r(e["value"]) for e in r["effects"]},
            "vue_pilier3_meur": {x["category"]: _r(x["value"]) for x in pillar3_view(r)["rows"]},
            "reconciliation_ok": r["reconciled"], "residuel_meur": _r(r["residual"]),
            "principaux_contributeurs": [
                {"contrepartie": c["cpty_id"], "secteur": c["sector"], "variation_meur": _r(c["delta"]),
                 "explication": c["explanation"]} for c in r["top_contributors"]],
            "analyses_existantes": [a["title"] + f" ({a['date']})" for a in past] or "aucune",
        }
        return {"tool": name, "title": "Variation du RWA crédit", "facts": facts,
                "display": {"kind": "variation", "result": r, "analyses": past}, "sql": None}

    if name == "detect_anomalies":
        a = detect_anomalies(b["s0"], b["s1"], top_n=5)
        facts = {"methode": a["method"], "positions_analysees": a["n_scored"], "alertes": [
            {"contrepartie": i["cpty_id"], "secteur": i["sector"], "score": _r(i["score"], 2),
             "ead_meur": _r(i["ead"]), "raison": i["reason"]} for i in a["items"]],
            "avertissement": "signaux à vérifier, pas des erreurs confirmées"}
        return {"tool": name, "title": "Contreparties atypiques", "facts": facts,
                "display": {"kind": "anomalies", "result": a}, "sql": None}

    if name == "leverage_ratio":
        lv = leverage_summary(b["series"])
        dd = explain_leverage(b["s0"], b["s1"], b["series"])
        past = find_analyses("leverage_ratio", lv["date"])
        facts = {
            "metrique": "Ratio de levier (illustratif)", "date": lv["date"],
            "levier_pct": _r(lv["leverage_pct"], 2), "mois_precedent_pct": _r(lv["previous_pct"], 2),
            "variation_pb": _r(lv["change_bp"], 1), "tier1_md_eur": _r(lv["tier1_bn"], 1),
            "tier1_variation": "inchangé entre M-1 et M",
            "variation_mois_pct": _r(lv["change_pct"], 1), "ecart_robuste_variation_mois": _r(lv["z_last"], 1),
            "variation_mois_aberrante": abs(lv["z_last"]) > 3.5,
            "mesure_exposition_md_eur": _r(lv["exposure_bn"], 1),
            "variation_exposition_md_eur": _r(dd["delta_exposure_bn"], 1),
            "origine_variation_pb": {e["label"]: _r(e["value"], 1) for e in dd["effects"]},
            "principaux_contributeurs": [
                {"contrepartie": c["cpty_id"], "variation_ead_meur": _r(c["delta_ead"]),
                 "contribution_pb": _r(c["contribution_bp"], 1), "explication": c["explanation"]}
                for c in dd["contributors"]],
            "historique_mois": 24,
            "anomalies_24_mois": [{"date": x["date"], "variation_pct": _r(x["change_pct"], 1), "analyse": x["analyse"]}
                                 for x in lv["anomalies"]] or "aucune",
            "analyses_existantes": [a["title"] + f" ({a['date']})" for a in past] or "aucune",
        }
        return {"tool": name, "title": "Ratio de levier", "facts": facts,
                "display": {"kind": "leverage", "result": lv, "drilldown": dd, "analyses": past}, "sql": None}

    if name == "define_term":
        d = define(str(args.get("term", "")))
        if d["found"]:
            facts = {"terme": d["term"], "definition": d["definition"], "precision": d["note"]}
        else:
            facts = {"terme_non_trouve": str(args.get("term", "")), "termes_disponibles": d["available"]}
        return {"tool": name, "title": "Définition", "facts": facts,
                "display": {"kind": "definition", "result": d}, "sql": None}

    if name == "query_metric":
        q = query_metric(ctx.con, args.get("name", ""), args.get("date"))
        facts = {"vue": q["title"], "date": q["date"], "unite": "M€", "lignes": q["rows"][:10]}
        return {"tool": name, "title": q["title"], "facts": facts,
                "display": {"kind": "query", "result": q}, "sql": q["sql"]}

    raise ValueError(f"Outil inconnu : {name}")


# --------------------------------------------------------------------------- Contrôle des chiffres
_NUM = re.compile(r"\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}(?:/\d{2,4})?")


def _collect(obj, out: list[float]):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(abs(float(obj)))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect(v, out)
    elif isinstance(obj, str):
        for m in _NUM.findall(_DATE.sub(" ", obj)):
            out.append(abs(_parse(m)))


def _parse(tok: str) -> float:
    return float(re.sub(r"[   ]", "", tok).replace(",", "."))


def check_numbers(text: str, facts: list[dict]) -> dict:
    """Vérifie que chaque nombre significatif du texte existe dans les résultats des outils."""
    known: list[float] = []
    for f in facts:
        _collect(f, known)
    unmatched = []
    for tok in _NUM.findall(_DATE.sub(" ", text)):
        dec = len(re.split(r"[.,]", tok)[1]) if re.search(r"[.,]\d", tok) else 0
        val = abs(_parse(tok))
        if dec == 0 and val < 10:
            continue  # petits entiers (comptes, rangs) : non contrôlés
        tol = 0.5 * 10 ** (-dec) * 1.001
        if not any(abs(val - k) <= tol for k in known):
            unmatched.append(tok.strip())
    return {"ok": not unmatched, "unmatched": unmatched}


# --------------------------------------------------------------------------- Mode démo (sans clé)
def _norm(q: str) -> str:
    return unicodedata.normalize("NFKD", q.lower()).encode("ascii", "ignore").decode()


def route_demo(question: str):
    q = _norm(question)
    if re.search(r"\bvar\b|value at risk|frtb|operationnel|stress|desk|actions", q):
        return None
    if re.search(r"c est quoi|qu est ce|definition|definis|signifie|veut dire", re.sub(r"[^a-z0-9]+", " ", q)):
        return ("define_term", {"term": question})
    if "levier" in q or "leverage" in q:
        return ("leverage_ratio", {})
    if any(k in q for k in ["anomal", "bizarre", "atypique", "outlier", "suspect", "etrange", "alerte"]):
        return ("detect_anomalies", {})
    if "secteur" in q:
        return ("query_metric", {"name": "rwa_by_sector"})
    if "par notation" in q or "par note" in q:
        return ("query_metric", {"name": "rwa_by_rating"})
    if "entite" in q or "perimetre" in q and "par" in q:
        return ("query_metric", {"name": "rwa_by_entity"})
    if any(k in q for k in ["plus grosses", "principales contreparties", "top "]):
        return ("query_metric", {"name": "top_counterparties"})
    if any(k in q for k in ["serie", "historique", "24 mois"]):
        return ("query_metric", {"name": "rwa_total_series"})
    if any(k in q for k in ["vari", "pourquoi", "explique", "chang", "evolu", "hausse", "baisse",
                            "rwa", "cascade", "walk", "perimetre"]):
        return ("explain_variation", {})
    return None


def render_demo(res: dict) -> str:
    f, kind = res["facts"], res["display"]["kind"]
    if kind == "definition":
        if "terme" in f:
            return f"**{f['terme']}** : {f['definition']}\n\n_{f['precision']}_"
        return ("Je n'ai pas cette définition dans mon glossaire. Termes disponibles : "
                + ", ".join(f["termes_disponibles"]) + ".")
    if kind == "variation":
        lines = [f"Le RWA crédit (illustratif) passe de {fr(f['rwa_precedent_meur'])} M€ à {fr(f['rwa_meur'])} M€, "
                 f"soit **{fr(f['variation_meur'], 0, True)} M€ ({fr(f['variation_pct'], 1, True)} %)**.", ""]
        for label, v in f["effets_meur"].items():
            if abs(v) >= 0.5:
                lines.append(f"- {label} : {fr(v, 0, True)} M€")
        head, bullets = lines[0], [l for l in lines[2:]]
        tail = ["Principal contributeur : " + _contrib(f["principaux_contributeurs"][0]) + "."]
        tail.insert(0, "Vue type Pilier 3 : " + ", ".join(f"{k} {fr(v, 0, True)} M€" for k, v in f["vue_pilier3_meur"].items()) + ".")
        if f["reconciliation_ok"]:
            tail.append("Contrôle : la somme des effets est égale à la variation totale.")
        tail += [t for t in [_past(f["analyses_existantes"])] if t]
        return head + "\n\n" + "\n".join(bullets) + "\n\n" + "  \n".join(tail)
    if kind == "anomalies":
        lines = ["Contreparties les plus atypiques sur la clôture (signaux à vérifier, pas des erreurs confirmées) :", ""]
        for a in f["alertes"][:4]:
            lines.append(f"- {a['contrepartie']} ({a['secteur']}), score {fr(a['score'], 2)} : {a['raison']}")
        return "\n".join(lines)
    if kind == "leverage":
        head = (f"Le ratio de levier (illustratif) est de **{fr(f['levier_pct'], 2)} %**, contre {fr(f['mois_precedent_pct'], 2)} % "
                f"le mois précédent ({fr(f['variation_pb'], 1, True)} pb). Tier 1 : {fr(f['tier1_md_eur'], 1)} Md€ (inchangé), "
                f"mesure d'exposition : {fr(f['mesure_exposition_md_eur'], 1)} Md€ "
                f"({fr(f['variation_exposition_md_eur'], 1, True)} Md€).")
        bullets = "\n".join(f"- {k} : {fr(v, 1, True)} pb" for k, v in f["origine_variation_pb"].items() if abs(v) >= 0.05)
        c = f["principaux_contributeurs"][0]
        tail = [f"Principal contributeur : {c['contrepartie']} ({fr(c['contribution_pb'], 1, True)} pb, "
                f"{c['explication'][0].lower() + c['explication'][1:]})."]
        verdict = "aberrante" if f["variation_mois_aberrante"] else "dans la norme"
        tail.append(f"La variation du mois ({fr(f['variation_mois_pct'], 1, True)} %, z = {fr(f['ecart_robuste_variation_mois'], 1, True)}) est {verdict}.")
        an = f["anomalies_24_mois"]
        if an == "aucune":
            tail.append("Aucune variation mensuelle aberrante sur 24 mois.")
        else:
            tail.append("Variations mensuelles aberrantes sur 24 mois : " + ", ".join(
                f"{fdate(x['date'])} ({fr(x['variation_pct'], 1, True)} %" + (", déjà analysée" if x["analyse"] != "aucune" else "")
                + ")" for x in an) + ".")
        return head + "\n\n**Origine de la variation du mois :**\n" + bullets + "\n\n" + "  \n".join(tail)
    rows = f["lignes"]
    lines = [f"{f['vue']} au {f['date']} (en M€) :", ""]
    for r in rows[:6]:
        vals = list(r.values())
        lines.append(f"- {vals[0]} : {fr(vals[-1])}")
    return "\n".join(lines)


def _contrib(c: dict) -> str:
    return f"{c['contrepartie']} ({fr(c['variation_meur'], 0, True)} M€, {c['explication'][0].lower() + c['explication'][1:]})"


def _past(a) -> str:
    return "" if a == "aucune" else "Analyse déjà réalisée : " + " ; ".join(a) + "."


# --------------------------------------------------------------------------- Orchestration
def answer(question: str, ctx: Ctx, api_key: str | None = None, model: str = "mistral-small-latest",
           timeout: int = 40) -> dict:
    """Retourne {text, results, mode, check, note}."""
    if api_key:
        try:
            return _answer_mistral(question, ctx, api_key, model, timeout)
        except Exception as exc:  # réseau, quota, modèle introuvable...
            out = _answer_demo(question, ctx)
            out["note"] = f"Mistral indisponible ({type(exc).__name__}) : réponse en mode démo."
            return out
    return _answer_demo(question, ctx)


def _answer_demo(question: str, ctx: Ctx) -> dict:
    routed = route_demo(question)
    if routed is None:
        return {"text": REFUSAL, "results": [], "mode": "demo", "check": None, "note": None}
    res = run_tool(routed[0], routed[1], ctx)
    text = render_demo(res)
    return {"text": text, "results": [res], "mode": "demo",
            "check": check_numbers(text, [res["facts"]]), "note": None}


def _chat(api_key, model, messages, tools, timeout):
    body = {"model": model, "messages": messages, "temperature": 0.1, "max_tokens": 500}
    if tools:
        body.update(tools=TOOLS, tool_choice="auto")
    r = requests.post(MISTRAL_URL, json=body, timeout=timeout,
                      headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


def _answer_mistral(question, ctx, api_key, model, timeout) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    msg = _chat(api_key, model, messages, True, timeout)
    calls = msg.get("tool_calls") or []
    if not calls:
        # Aucun outil appelé : on refuse plutôt que de laisser le modèle répondre de mémoire.
        return {"text": REFUSAL, "results": [], "mode": "mistral", "check": None, "note": None}

    messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
    results = []
    for c in calls[:3]:
        fn = c["function"]
        try:
            args = json.loads(fn.get("arguments") or "{}")
            res = run_tool(fn["name"], args, ctx)
        except Exception as exc:
            res = {"tool": fn["name"], "title": "Erreur", "facts": {"erreur": str(exc)},
                   "display": {"kind": "error"}, "sql": None}
        results.append(res)
        messages.append({"role": "tool", "name": fn["name"], "tool_call_id": c["id"],
                         "content": json.dumps(res["facts"], ensure_ascii=False)})
    final = _chat(api_key, model, messages, False, timeout)
    text = (final.get("content") or "").strip()
    return {"text": text, "results": results, "mode": "mistral",
            "check": check_numbers(text, [r["facts"] for r in results]), "note": None}
