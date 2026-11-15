"""Tests : réconciliation, événements injectés retrouvés, anomalies, lecture seule SQL."""
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvency.data import D0, D1, make_bundle  # noqa: E402
from solvency.data import EXPOSURE_FACTOR  # noqa: E402
from solvency.engine import (  # noqa: E402
    build_db, detect_anomalies, detect_series_anomalies, enrich, explain_leverage, explain_variation,
    find_analyses, leverage_summary, query_metric,
)


@pytest.fixture(scope="module")
def bundle():
    return make_bundle()


@pytest.fixture(scope="module")
def result(bundle):
    return explain_variation(bundle["s0"], bundle["s1"])


def effect(result, key):
    return next(e["value"] for e in result["effects"] if e["key"] == key)


def test_reconciliation(result):
    """La somme des effets doit être égale à la variation totale."""
    assert result["reconciled"]
    assert abs(sum(e["value"] for e in result["effects"]) - result["delta"]) < 1e-6


def test_data_is_reproducible():
    a, b = make_bundle(), make_bundle()
    assert a["s1"].equals(b["s1"])


def test_perimeter_effect_equals_new_york_rwa(bundle, result):
    s1 = enrich(bundle["s1"])
    ny = s1[(s1["entity"] == "Périmètre acquis")]["rwa"].sum()
    assert effect(result, "perimeter") == pytest.approx(ny, rel=1e-9)


def test_exits_and_new_deals(bundle, result):
    t = bundle["truth"]
    s0, s1 = enrich(bundle["s0"]), enrich(bundle["s1"])
    exits = s0[s0["cpty_id"].isin(t["exits"])]["rwa"].sum()
    assert effect(result, "exit") == pytest.approx(-exits, rel=1e-9)
    assert effect(result, "new") >= s1[s1["cpty_id"] == t["new_large"]]["rwa"].iloc[0]


def test_rating_effect_equals_the_injected_rating_changes(bundle, result):
    a, b = enrich(bundle["s0"]), enrich(bundle["s1"])
    expected = 0.0
    for cid in bundle["truth"]["downgraded"]:
        r0 = a[a["cpty_id"] == cid].iloc[0]
        r1 = b[b["cpty_id"] == cid].iloc[0]
        expected += r1["ead"] * (r1["rw"] - r0["rw"])
    for cid in bundle["truth"]["upgraded"]:
        r0 = a[a["cpty_id"] == cid].iloc[0]
        r1 = b[b["cpty_id"] == cid].iloc[0]
        expected += r1["ead"] * (r1["rw"] - r0["rw"])
    # + la dégradation anormale du souverain CP0777 (AA -> CCC)
    r0 = a[a["cpty_id"] == "CP0777"].iloc[0]
    r1 = b[b["cpty_id"] == "CP0777"].iloc[0]
    expected += r1["ead"] * (r1["rw"] - r0["rw"])
    assert effect(result, "rating") == pytest.approx(expected, rel=1e-9)
    assert effect(result, "rating") > 0


def test_residual_is_the_injected_overlay(bundle, result):
    assert result["residual"] == pytest.approx(bundle["truth"]["overlay_amount"], abs=1e-9)


def test_large_new_deal_is_a_top_contributor(bundle, result):
    assert bundle["truth"]["new_large"] in [c["cpty_id"] for c in result["top_contributors"]][:3]


def test_pillar3_view_sums_to_total(result):
    from solvency.engine import pillar3_view
    v = pillar3_view(result)
    assert v["start"] + sum(r["value"] for r in v["rows"]) == pytest.approx(v["end"])
    assert [r["category"] for r in v["rows"]] == ["Taille des actifs", "Qualité des actifs",
                                                  "Acquisitions et cessions", "Autres"]


def test_injected_anomalies_are_found(bundle):
    t = bundle["truth"]
    items = detect_anomalies(bundle["s0"], bundle["s1"], top_n=12)["items"]
    ids = [i["cpty_id"] for i in items]
    assert ids[:2] == [t["anomaly_ead"], t["anomaly_rating"]]          # les deux anomalies injectées en tête
    for must in t["downgraded"]:
        assert must in ids, f"{must} non détectée"
    assert all(i["reason"] for i in items)


def test_series_spike_detected(bundle):
    found = [a["date"] for a in detect_series_anomalies(bundle["series"], "leverage")]
    assert bundle["truth"]["series_spike_date"] in found


def test_series_last_values_match_snapshots(bundle):
    s = bundle["series"]
    assert s["rwa_total"].iloc[-1] == pytest.approx(enrich(bundle["s1"])["rwa"].sum())
    assert s["rwa_total"].iloc[-2] == pytest.approx(enrich(bundle["s0"])["rwa"].sum())


def test_sql_is_read_only_and_whitelisted(bundle):
    con = build_db(bundle)
    with pytest.raises(sqlite3.OperationalError):
        con.execute("DELETE FROM exposures")
    with pytest.raises(ValueError):
        query_metric(con, "DROP TABLE exposures")
    with pytest.raises(ValueError):
        query_metric(con, "rwa_by_sector", date="2020-01-01")
    out = query_metric(con, "rwa_by_sector", D1)
    assert out["rows"] and "SELECT" in out["sql"] and D1 in out["sql"]


def test_no_removed_columns(bundle):
    assert not {"pd", "lgd", "maturity", "country", "currency"} & set(bundle["s1"].columns)


def test_totals_by_sector_match_total(bundle, result):
    con = build_db(bundle)
    rows = query_metric(con, "rwa_by_sector", D0)["rows"]
    assert sum(r["rwa_meur"] for r in rows) == pytest.approx(result["total0"], abs=len(rows))


def test_history_lookup():
    assert find_analyses("leverage_ratio", "2026-03-31")                   # pic de fin de trimestre déjà analysé
    assert not find_analyses("leverage_ratio", D1)                         # rien sur la variation du mois
    assert not find_analyses("rwa_credit", D1)


def test_leverage_is_linked_to_counterparties(bundle):
    s = bundle["series"]
    ead1 = bundle["s1"].loc[bundle["s1"]["in_scope"], "ead"].sum()
    assert s["exposure"].iloc[-1] == pytest.approx(EXPOSURE_FACTOR * ead1 / 1000)
    assert s["tier1"].iloc[-1] == pytest.approx(s["tier1"].iloc[-2])       # Tier 1 inchangé à J
    # ordres de grandeur proches d'une grande banque de financement et d'investissement
    assert 0.12 < s["tier1"].iloc[-1] * 1000 / s["rwa_total"].iloc[-1] < 0.18       # Tier 1 / RWA
    assert 0.20 < s["rwa_total"].iloc[-1] / 1000 / s["exposure"].iloc[-1] < 0.30    # RWA / mesure d'exposition
    assert 3.0 < s["leverage"].iloc[-1] < 4.5


def test_leverage_history_flags_the_quarter_end_spike_and_is_explained(bundle):
    lv = leverage_summary(bundle["series"])
    by_date = {a["date"]: a for a in lv["anomalies"]}
    spike = bundle["truth"]["series_spike_date"]
    assert spike in by_date and by_date[spike]["analyse"] != "aucune"
    assert D1 not in by_date and abs(lv["z_last"]) < 3.5          # la variation du mois est dans la norme
    assert len(bundle["series"]) == 24

    dd = explain_leverage(bundle["s0"], bundle["s1"], bundle["series"], top_n=5)
    assert dd["reconciled"]
    assert sum(e["value"] for e in dd["effects"]) == pytest.approx(dd["delta"], abs=1e-9)
    assert dd["delta"] == pytest.approx(lv["change_bp"], abs=1e-9)
    ny = enrich(bundle["s1"])
    ny_ead = ny[ny["entity"] == "Périmètre acquis"]["ead"].sum()
    perim = next(e for e in dd["effects"] if e["key"] == "perimeter")
    assert perim["exposure_bn"] == pytest.approx(EXPOSURE_FACTOR * ny_ead / 1000)
    assert dd["delta_exposure_bn"] > 0 and dd["delta"] < 0          # plus d'exposition, moins de levier
    assert bundle["truth"]["new_large"] in [c["cpty_id"] for c in dd["contributors"]]
    assert {e["entity"] for e in dd["by_entity"]} >= {"Périmètre acquis", "Périmètre existant"}


def test_rwa_series_flags_the_quarter_end_spike(bundle):
    from solvency.engine import rwa_series_summary
    rs = rwa_series_summary(bundle["series"])
    assert bundle["truth"]["series_spike_date"] in [a["date"] for a in rs["anomalies"]]
    assert D1 not in [a["date"] for a in rs["anomalies"]]       # la hausse du mois (périmètre) reste dans la norme
