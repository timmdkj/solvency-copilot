"""Solvency Metrics Copilot : prototype illustratif (données synthétiques)."""
from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from solvency import llm
from solvency.data import D0, D1, make_bundle
from solvency.engine import (build_db, detect_anomalies, explain_leverage, explain_variation, fdate, find_analyses, fr,
                             leverage_summary, pillar3_view, rwa_series_summary)

NAVY, TERRACOTTA, TEAL, GREY = "#1F3A5F", "#B5533C", "#3C7A89", "#8A94A3"
MAX_LLM_MESSAGES = 10

st.set_page_config(page_title="Solvency Metrics Copilot", page_icon=None, layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1180px;}
      h1 {font-size: 1.7rem !important; font-weight: 650; letter-spacing: -0.01em;}
      h2, h3 {font-weight: 600;}
      .tag {display:inline-block; padding:2px 10px; border:1px solid #C9D1DC; border-radius:999px;
            font-size:0.78rem; color:#4A5568; margin-right:6px;}
      [data-testid="stMetricValue"] {font-size: 1.6rem;}
      footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- Config et données
def secret(name: str, default=None):
    try:
        return st.secrets.get(name, os.environ.get(name, default))
    except Exception:
        return os.environ.get(name, default)


@st.cache_resource(show_spinner=False)
def load():
    bundle = make_bundle()
    return bundle, build_db(bundle)


@st.cache_data(show_spinner=False)
def variation_result():
    bundle, _ = load()
    return explain_variation(bundle["s0"], bundle["s1"], top_n=8)


@st.cache_data(show_spinner=False)
def anomalies_result():
    bundle, _ = load()
    return detect_anomalies(bundle["s0"], bundle["s1"], top_n=8)


def check_password() -> bool:
    pwd = secret("APP_PASSWORD")
    if not pwd or st.session_state.get("auth"):
        return True
    st.title("Solvency Metrics Copilot")
    entered = st.text_input("Mot de passe", type="password")
    if entered:
        if entered == pwd:
            st.session_state["auth"] = True
            st.rerun()
        st.error("Mot de passe incorrect.")
    return False


# --------------------------------------------------------------------------- Graphiques
def waterfall(result: dict, unit: str = "M€", nd: int = 0) -> go.Figure:
    eff = [e for e in result["effects"] if abs(e["value"]) >= 0.5 * 10 ** (-nd)]
    labels = [e["label"] for e in eff] + ["Variation totale"]
    values = [e["value"] for e in eff] + [result["delta"]]
    fig = go.Figure(go.Waterfall(
        x=labels, y=values, measure=["relative"] * len(eff) + ["total"],
        text=[fr(v, nd, True) for v in values], textposition="outside",
        increasing={"marker": {"color": TERRACOTTA}}, decreasing={"marker": {"color": TEAL}},
        totals={"marker": {"color": NAVY}}, connector={"line": {"color": "#C9D1DC", "width": 1}},
        hovertemplate="%{x}<br>%{y:+,." + str(nd) + "f} " + unit + "<extra></extra>",
    ))
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
        yaxis_title=unit, plot_bgcolor="white", paper_bgcolor="white", separators=", ",
        font=dict(family="sans-serif", size=12, color="#1B2430"),
    )
    fig.update_yaxes(gridcolor="#EEF1F5", zerolinecolor="#C9D1DC")
    fig.update_xaxes(tickangle=-25)
    return fig


def line_chart(df: pd.DataFrame, col: str, title: str, flagged: list[dict]) -> go.Figure:
    fig = go.Figure(go.Scatter(x=df["date"], y=df[col], mode="lines", line=dict(color=NAVY, width=2),
                               hovertemplate="%{x|%d/%m/%Y}<br>%{y:,.2f}<extra></extra>"))
    if flagged:
        fx = pd.to_datetime([f["date"] for f in flagged])
        fig.add_trace(go.Scatter(x=fx, y=df.set_index("date").loc[fx, col], mode="markers",
                                 marker=dict(color=TERRACOTTA, size=9), name="Anomalie",
                                 hovertemplate="%{x|%d/%m/%Y}<br>%{y:,.2f}<extra>Anomalie</extra>"))
    fig.update_layout(height=290, margin=dict(l=10, r=10, t=30, b=10), showlegend=False, title=dict(text=title, font=dict(size=14)),
                      plot_bgcolor="white", paper_bgcolor="white", separators=", ",
                      font=dict(family="sans-serif", size=12, color="#1B2430"))
    fig.update_yaxes(gridcolor="#EEF1F5")
    fig.update_xaxes(tickformat="%m/%Y")
    return fig


# --------------------------------------------------------------------------- Blocs d'affichage
def effects_table(result: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "Effet": [e["label"] for e in result["effects"]] + ["Variation totale"],
        "Catégorie type Pilier 3": [e["pillar3"] for e in result["effects"]] + [""],
        "M€": [fr(e["value"], 0, True) for e in result["effects"]] + [fr(result["delta"], 0, True)],
    })


def pillar3_table(result: dict) -> pd.DataFrame:
    v = pillar3_view(result)
    rows = [{"Flux": f"RWA au {fdate(result['date0'])}", "M€": fr(v["start"])}]
    rows += [{"Flux": r["category"], "M€": fr(r["value"], 0, True)} for r in v["rows"]]
    rows += [{"Flux": f"RWA au {fdate(result['date1'])}", "M€": fr(v["end"])}]
    return pd.DataFrame(rows)


def contributors_table(result: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "Contrepartie": c["cpty_id"], "Secteur": c["sector"],
        "Variation (M€)": fr(c["delta"], 0, True), "Explication": c["explanation"],
    } for c in result["top_contributors"]])


def anomalies_table(res: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "Contrepartie": i["cpty_id"], "Secteur": i["sector"], "Notation": i["rating"], "EAD (M€)": fr(i["ead"]),
        "Score": fr(i["score"], 2), "Raison": i["reason"],
    } for i in res["items"]])


def show_analyses(analyses: list[dict]):
    if analyses:
        for a in analyses:
            st.caption(f"Analyse déjà réalisée : {a['title']} ({a['date']}, {a['author']})")
    else:
        pass


def show_leverage(lv: dict, dd: dict, key: str):
    bundle, _ = load()
    verdict = "aberrante" if abs(lv["z_last"]) > 3.5 else "dans la norme"
    st.caption(f"Dernier mois ({fdate(D0)} → {fdate(D1)}) : le levier varie de {fr(lv['change_pct'], 1, True)} %. "
               f"Écart par rapport aux variations habituelles : z = {fr(lv['z_last'], 1, True)}, {verdict} (alerte au-delà de ±3,5). "
               "Les points orange sont les clôtures aberrantes de l'historique.")
    st.plotly_chart(line_chart(bundle["series"], "leverage", "Ratio de levier (%), 24 clôtures mensuelles", lv["anomalies"]),
                    use_container_width=True, key=f"lv-{key}")
    if lv["anomalies"]:
        st.dataframe(pd.DataFrame([{
            "Clôture": fdate(x["date"]), "Variation vs mois précédent": f"{fr(x['change_pct'], 1, True)} %",
            "Analyse existante": x["analyse"]} for x in lv["anomalies"]]),
            hide_index=True, use_container_width=True)
    st.markdown(f"**Origine de la variation du mois : {fr(dd['delta'], 1, True)} pb**")
    st.plotly_chart(waterfall(dd, unit="pb", nd=1), use_container_width=True, key=f"lvwf-{key}")
    left, right = st.columns(2)
    with left:
        st.dataframe(pd.DataFrame([{
            "Périmètre": e["entity"], "Exposition (Md€)": fr(e["exposure_bn"], 2, True),
            "Contribution (pb)": fr(e["value"], 1, True)} for e in dd["by_entity"]]),
            hide_index=True, use_container_width=True)
    with right:
        st.dataframe(pd.DataFrame([{
            "Contrepartie": c["cpty_id"], "Contribution (pb)": fr(c["contribution_bp"], 1, True),
            "Explication": c["explanation"]} for c in dd["contributors"]]),
            hide_index=True, use_container_width=True)
    st.caption("Tier 1 supposé inchangé entre M-1 et M (simplification) : la variation vient de la mesure d'exposition. "
               "Contributions réparties au prorata de la variation d'exposition.")


def render_evidence(res: dict, key: str):
    d = res["display"]
    if d["kind"] == "variation":
        r = d["result"]
        st.plotly_chart(waterfall(r), use_container_width=True, key=f"wf-{key}")
        st.dataframe(contributors_table(r), hide_index=True, use_container_width=True)
        st.caption(f"Réconciliation : somme des effets = variation totale (écart {r['reconciliation_gap']:.1e} M€).")
        show_analyses(d["analyses"])
    elif d["kind"] == "anomalies":
        st.dataframe(anomalies_table(d["result"]), hide_index=True, use_container_width=True)
        st.caption(f"Méthode : {d['result']['method']} sur {d['result']['n_scored']} positions.")
    elif d["kind"] == "leverage":
        show_leverage(d["result"], d["drilldown"], key)
        show_analyses(d["analyses"])
    elif d["kind"] == "query":
        q = d["result"]
        st.dataframe(pd.DataFrame(q["rows"]), hide_index=True, use_container_width=True)
    if res.get("sql"):
        st.code(res["sql"], language="sql")


# --------------------------------------------------------------------------- Application
if not check_password():
    st.stop()

bundle, con = load()
ctx = llm.Ctx(bundle=bundle, con=con)
api_key = secret("MISTRAL_API_KEY")
model = secret("MODEL", "mistral-small-latest")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.llm_count = 0

EXAMPLES = [
    "Pourquoi le RWA a-t-il varié ce mois-ci ?",
    "Y a-t-il des contreparties atypiques ?",
    "Quel est le ratio de levier ?",
    "RWA par secteur",
    "C'est quoi l'effet de périmètre ?",
    "Quelle est la VaR de la desk Actions ?",
]

with st.sidebar:
    st.markdown("### Solvency Metrics Copilot")
    st.caption("Prototype illustratif.")
    using_llm = bool(api_key) and st.session_state.llm_count < MAX_LLM_MESSAGES
    st.markdown(f"<span class='tag'>{'Mistral : ' + model if using_llm else 'Mode démo'}</span>"
                f"<span class='tag'>Clôture M = {fdate(D1)}</span>", unsafe_allow_html=True)
    st.markdown("**Questions d'exemple**")
    clicked = None
    for i, ex in enumerate(EXAMPLES):
        if st.button(ex, key=f"ex{i}", use_container_width=True):
            clicked = ex
    with st.expander("À propos"):
        st.markdown(
            "- Le **calcul est déterministe** (Python, SQL).\n"
            "- Le modèle de langage **choisit l'outil et rédige** : il ne calcule rien.\n"
            "- Chaque chiffre cité est **recontrôlé** contre les résultats.\n"
            "- Les requêtes SQL sont **prédéfinies et en lecture seule**.\n"
            "- Hors périmètre, l'assistant **refuse** plutôt que d'inventer.\n"
            "- Les définitions viennent d'un **petit glossaire** du prototype."
        )
    if st.button("Effacer la conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("Solvency Metrics Copilot")
st.caption("Agent IA d’analyse des métriques de solvabilité : variations de RWA, anomalies, ratio de levier.")

tab_chat, tab_var, tab_alert = st.tabs(["Assistant", "Variation du RWA", "Alertes"])

with tab_chat:
    if not st.session_state.messages:
        st.info("Posez une question ou choisissez un exemple dans le menu de gauche.")
    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"], avatar=":material/person:" if m["role"] == "user" else ":material/analytics:"):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                if m.get("note"):
                    st.caption(m["note"])
                for j, res in enumerate(m.get("results", [])):
                    if res["display"]["kind"] == "definition":
                        continue
                    with st.expander("Preuves : données, requête et contrôles", expanded=(i == len(st.session_state.messages) - 1)):
                        render_evidence(res, f"{i}-{j}")
                chk = m.get("check")
                if chk is not None:
                    if chk["ok"]:
                        st.caption("Contrôle des chiffres : tous les nombres cités figurent dans les résultats calculés.")
                    else:
                        st.warning("Contrôle des chiffres : à vérifier, non retrouvés dans les résultats : "
                                   + ", ".join(chk["unmatched"]))
    question = st.chat_input("Votre question sur les métriques…") or clicked
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        use_key = api_key if st.session_state.llm_count < MAX_LLM_MESSAGES else None
        with st.spinner("Calcul en cours…"):
            out = llm.answer(question, ctx, api_key=use_key, model=model)
        if out["mode"] == "mistral":
            st.session_state.llm_count += 1
        st.session_state.messages.append({"role": "assistant", "content": out["text"], "results": out["results"],
                                          "check": out["check"], "note": out["note"]})
        st.rerun()

with tab_var:
    r = variation_result()
    c1, c2, c3 = st.columns(3)
    c1.metric(f"RWA crédit, clôture {fdate(D0)}", f"{fr(r['total0'])} M€")
    c2.metric(f"RWA crédit, clôture {fdate(D1)}", f"{fr(r['total1'])} M€", f"{fr(r['delta'], 0, True)} M€ ({fr(r['delta_pct'], 1, True)} %)", delta_color="off")
    c3.metric("Résiduel", f"{fr(r['residual'], 0, True)} M€")
    bundle, _ = load()
    rs = rwa_series_summary(bundle["series"])
    verdict = "aberrante" if abs(rs["z_last"]) > 3.5 else "dans la norme"
    st.caption(f"Dernier mois ({fdate(D0)} → {fdate(D1)}) : le RWA varie de {fr(rs['change_pct'], 1, True)} %. "
               f"Écart par rapport aux variations habituelles : z = {fr(rs['z_last'], 1, True)}, {verdict} (alerte au-delà de ±3,5). "
               "Les points orange sont les clôtures aberrantes de l'historique.")
    st.plotly_chart(line_chart(bundle["series"].assign(rwa_md=bundle["series"]["rwa_total"] / 1000), "rwa_md",
                               "RWA total (Md€), 24 clôtures mensuelles", rs["anomalies"]),
                    use_container_width=True, key="rwa-series")
    if rs["anomalies"]:
        st.dataframe(pd.DataFrame([{
            "Clôture": fdate(x["date"]), "Variation vs mois précédent": f"{fr(x['change_pct'], 1, True)} %",
            "Analyse existante": x["analyse"]} for x in rs["anomalies"]]),
            hide_index=True, use_container_width=True)
    st.markdown("#### Décomposition de la variation")
    st.plotly_chart(waterfall(r), use_container_width=True, key="wf-main")
    st.markdown("#### Principaux contributeurs")
    st.dataframe(contributors_table(r), hide_index=True, use_container_width=True)
    with st.expander("Tableau des effets et correspondance avec le Pilier 3"):
        st.dataframe(effects_table(r), hide_index=True, use_container_width=True)
        st.markdown("**Présentation type tableau de flux du Pilier 3 (CR8)**")
        st.dataframe(pillar3_table(r), hide_index=True, use_container_width=True)
        st.caption("Les publications officielles sont trimestrielles et comportent aussi des lignes que ce prototype "
                   "ne modélise pas (mises à jour de modèles, méthodologie, change).")
    st.caption(f"Réconciliation : somme des effets = variation totale (écart {r['reconciliation_gap']:.1e} M€). "
               "Ordre de la décomposition : périmètre, nouvelles opérations, sorties, encours, notation, résiduel.")
    show_analyses(find_analyses("rwa_credit", D1))

with tab_alert:
    bundle, _ = load()
    lv = leverage_summary(bundle["series"])
    dd = explain_leverage(bundle["s0"], bundle["s1"], bundle["series"])
    st.markdown("#### Ratio de levier")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Ratio de levier, clôture {fdate(D1)}", f"{fr(lv['leverage_pct'], 2)} %", f"{fr(lv['change_bp'], 1, True)} pb vs {fdate(D0)}", delta_color="off")
    c2.metric("Tier 1 (inchangé)", f"{fr(lv['tier1_bn'], 1)} Md€")
    c3.metric("Mesure d'exposition", f"{fr(lv['exposure_bn'], 1)} Md€",
              f"{fr(dd['delta_exposure_bn'], 1, True)} Md€", delta_color="off")
    show_leverage(lv, dd, "alert")
    st.markdown("#### Contreparties atypiques")
    a = anomalies_result()
    st.dataframe(anomalies_table(a), hide_index=True, use_container_width=True)
    st.caption(f"{a['method']} sur {a['n_scored']} positions, centré sur les variations depuis la clôture précédente, "
               "hors mouvements communs à tout un secteur. Ce sont des signaux à vérifier, pas des erreurs confirmées.")
