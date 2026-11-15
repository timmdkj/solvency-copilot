"""Tests de la couche conversationnelle : routage démo, refus, contrôle des chiffres, flux Mistral simulé."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvency import llm  # noqa: E402
from solvency.data import make_bundle  # noqa: E402
from solvency.engine import build_db  # noqa: E402


@pytest.fixture(scope="module")
def ctx():
    b = make_bundle()
    return llm.Ctx(bundle=b, con=build_db(b))


@pytest.mark.parametrize("q,tool", [
    ("Pourquoi le RWA a varié aujourd'hui ?", "explain_variation"),
    ("Et le ratio de levier ?", "leverage_ratio"),
    ("Y a des choses bizarres sur les contreparties ?", "detect_anomalies"),
    ("RWA par secteur", "query_metric"),
    ("RWA par notation", "query_metric"),
    ("C'est quoi l'effet de périmètre ?", "define_term"),
    ("quelles sont les plus grosses contreparties", "query_metric"),
])
def test_demo_routing(ctx, q, tool):
    out = llm.answer(q, ctx)
    assert out["mode"] == "demo" and out["results"][0]["tool"] == tool
    assert out["check"]["ok"], out["check"]


def test_definition_uses_glossary_and_unknown_term_lists_available(ctx):
    ok = llm.answer("C'est quoi l'effet de périmètre ?", ctx)
    assert "périmètre de consolidation" in ok["text"] and ok["check"]["ok"]
    unknown = llm.answer("C'est quoi l'effet de ratio ?", ctx)
    assert "Termes disponibles" in unknown["text"]


def test_leverage_answer_explains_the_origin(ctx):
    out = llm.answer("Pourquoi le ratio de levier a baissé ?", ctx)
    assert out["results"][0]["tool"] == "leverage_ratio" and out["check"]["ok"], out["check"]
    assert "Origine de la variation du mois" in out["text"] and "périmètre" in out["text"].lower()
    assert "dans la norme" in out["text"] and "déjà analysée" in out["text"]


def test_out_of_scope_is_refused(ctx):
    out = llm.answer("Quelle est la VaR de la desk Actions ?", ctx)
    assert out["text"] == llm.REFUSAL and not out["results"]


def test_variation_answer_has_pillar3_view(ctx):
    out = llm.answer("Pourquoi le RWA a varié ce mois-ci ?", ctx)
    assert "Vue type Pilier 3" in out["text"] and "Taille des actifs" in out["text"] and out["check"]["ok"]


def test_demo_text_contains_the_computed_numbers(ctx):
    out = llm.answer("Pourquoi le RWA a varié ?", ctx)
    f = out["results"][0]["facts"]
    assert llm.fr(f["rwa_meur"]) in out["text"]
    assert "somme des effets" in out["text"]


def test_number_check_catches_invented_figures():
    facts = [{"variation_meur": 1506, "variation_pct": 4.2}]
    assert llm.check_numbers("Hausse de +1 506 M€ (+4,2 %), le 2026-11-13.", facts)["ok"]
    bad = llm.check_numbers("Hausse de 1 800 M€.", facts)
    assert not bad["ok"] and bad["unmatched"] == ["1 800"]


def test_mistral_flow_with_tool_call(ctx, monkeypatch):
    calls = []

    class Resp:
        def __init__(self, msg): self._m = msg
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": self._m}]}

    def fake_post(url, json=None, **kw):
        calls.append(json)
        if len(calls) == 1:
            return Resp({"content": "", "tool_calls": [{"id": "t1", "function": {
                "name": "leverage_ratio", "arguments": "{}"}}]})
        tool_msg = [m for m in json["messages"] if m["role"] == "tool"][0]
        facts = __import__("json").loads(tool_msg["content"])
        return Resp({"content": f"Le ratio de levier est de {str(facts['levier_pct']).replace('.', ',')} %."})

    monkeypatch.setattr(llm.requests, "post", fake_post)
    out = llm.answer("Et le ratio de levier ?", ctx, api_key="x")
    assert out["mode"] == "mistral" and out["check"]["ok"] and len(calls) == 2


def test_mistral_without_tool_call_is_refused(ctx, monkeypatch):
    class Resp:
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": "La VaR est de 12 M€."}}]}
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: Resp())
    out = llm.answer("Quelle est la VaR ?", ctx, api_key="x")
    assert out["text"] == llm.REFUSAL


def test_mistral_failure_falls_back_to_demo(ctx, monkeypatch):
    def boom(*a, **k): raise llm.requests.ConnectionError("réseau")
    monkeypatch.setattr(llm.requests, "post", boom)
    out = llm.answer("Et le ratio de levier ?", ctx, api_key="x")
    assert out["mode"] == "demo" and out["note"] and out["results"]
