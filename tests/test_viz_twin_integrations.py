"""Bridges to other 3-D viewers (Rerun, Plotly, Trame) and digital-twin standards (AAS, DTDL, NGSI-LD, Ditto,
OPC UA)."""
import asyncio
import io
import json
import re
import socket

import numpy as np
import pytest

from pinneapple_systems.digital_twin.standards import (
    OpcUaTwinServer,
    TwinDescription,
    TwinModel,
    TwinProperty,
    to_aas,
    to_ditto_thing,
    to_dtdl,
    to_ngsi_ld,
)


@pytest.fixture
def twin():
    return TwinDescription(
        "urn:pinneapple:twin:pump-17", "Pump 17", "CentrifugalPump", "Cooling-loop pump", "ACME",
        [TwinProperty("rated_flow", 0.02, "m3/s"), TwinProperty("speed", 1450.0, "rpm", "telemetry"),
         TwinProperty("head_mean", 31.5, "m", "state", "estimated head"),
         TwinProperty("setpoint", 1450.0, "rpm", "parameter", writable=True)],
        [TwinModel("pump_surrogate", "FNO", "VALID", 0.97), TwinModel("laminar_fit", "closed form", "VIOLATED", 0.0,
                                                                     ["laminar"], "Regime change")],
        [("feeds", "urn:pinneapple:twin:hx-3")])


def test_property_validation():
    with pytest.raises(ValueError):
        TwinProperty("1bad", 1.0)
    with pytest.raises(ValueError):
        TwinProperty("ok", 1.0, kind="weird")


def test_from_snapshot_and_audit():
    d = TwinDescription.from_snapshot("urn:x:t1", "T1", {"fields": {"T": np.array([300.0, 310.0]), "p": 101325.0}},
                                      units={"T": "K", "p": "Pa"})
    names = {p.name: p for p in d.properties}
    assert names["T_mean"].value == pytest.approx(305.0) and names["T_mean"].kind == "state"
    assert names["p"].kind == "telemetry" and names["p"].unit == "Pa"
    from pinneapple_physics.physics_audit import (
        ModelProfile,
        attribute_failure,
        audit_assumptions,
        laminar,
    )

    a = audit_assumptions(ModelProfile("lam", "", (), (laminar(),)), {"Re": 6000.0})
    m = TwinModel.from_audit(a, "closed form", attribute_failure(audit=a, error=0.5))
    assert m.status == "VIOLATED" and m.violated == ["laminar"] and "Regime change" in m.probable_cause


def test_aas_roundtrip(twin):
    pytest.importorskip("basyx")
    from basyx.aas import model
    from basyx.aas.adapter import json as aasjson

    text = to_aas(twin)
    d = json.loads(text)
    assert len(d["assetAdministrationShells"]) == 1 and len(d["submodels"]) == 4
    store = aasjson.read_aas_json_file(io.StringIO(text))           # BaSyx reads back what it wrote
    sms = {o.id_short: o for o in store if isinstance(o, model.Submodel)}
    assert set(sms) == {"Identification", "TechnicalData", "OperationalData", "SimulationModels"}
    tech = {e.id_short: e.value for e in sms["TechnicalData"].submodel_element}
    assert tech["rated_flow"] == pytest.approx(0.02)
    lam = sms["SimulationModels"].get_referable("laminar_fit")
    assert {e.id_short: e.value for e in lam.value}["ViolatedAssumptions"] == "laminar"


def test_dtdl_ngsi_ditto(twin):
    dt = to_dtdl(twin)
    assert dt["@context"] == "dtmi:dtdl:context;3" and dt["@type"] == "Interface"
    assert re.fullmatch(r"dtmi:[A-Za-z](?:[A-Za-z0-9_]*[A-Za-z0-9])?(?::[A-Za-z](?:[A-Za-z0-9_]*[A-Za-z0-9])?)*;[1-9][0-9]*",
                        dt["@id"])
    kinds = {c["name"]: c["@type"] for c in dt["contents"]}
    assert kinds["speed"] == "Telemetry" and kinds["rated_flow"] == "Property" and kinds["feeds"] == "Relationship"
    assert all(len(c["name"]) <= 64 and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", c["name"]) for c in dt["contents"])
    ng = to_ngsi_ld(twin, observed_at="2026-10-11T12:00:00Z")
    assert ng["id"].startswith("urn:ngsi-ld:") and ng["speed"]["unitText"] == "rpm"
    assert ng["speed"]["observedAt"] and ng["feeds"] == {"type": "Relationship", "object": "urn:pinneapple:twin:hx-3"}
    assert ng["validity_laminar_fit"]["value"] == "VIOLATED"
    th = to_ditto_thing(twin)
    assert th["thingId"] == "org.pinneapple:Pump-17"
    assert th["features"]["telemetry"]["properties"]["speed"]["value"] == 1450.0
    assert th["features"]["models"]["properties"]["laminar_fit"]["violated"] == ["laminar"]


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_opcua_server_read_and_update(twin):
    pytest.importorskip("asyncua")
    from asyncua import Client

    url = f"opc.tcp://127.0.0.1:{_free_port()}/pinneapple/"
    srv = OpcUaTwinServer(twin, url).start()
    try:
        srv.update("speed", 1500.0)

        async def read():
            async with Client(url) as c:
                return (await c.get_node(srv.node_ids["speed"]).read_value(),
                        await c.get_node(srv.node_ids["validity_laminar_fit"]).read_value())
        assert asyncio.run(read()) == (1500.0, "VIOLATED")
        with pytest.raises(KeyError):
            srv.update("nope", 1)
    finally:
        srv.stop()


def _scene():
    from pinneapple_tools.visualization.studio.scene import Scene

    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    F = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    sc = Scene.from_arrays(V, F, name="tet")
    sc.surfaces[0].fields["p"] = np.array([0.0, 1.0, 2.0, 3.0])
    sc.add_lines([np.array([[0, 0, 0], [0.5, 0.5, 0.5]])])
    return sc


def test_rerun_bridge(tmp_path):
    pytest.importorskip("rerun")
    from pinneapple_analysis.proof import ProofGraph, symbolic_identity
    from pinneapple_tools.visualization.integrations.rerun_bridge import RerunLogger

    path = tmp_path / "run.rrd"
    log = RerunLogger("pinneapple-test", path=str(path))
    log.scene(_scene(), field="p")
    log.points("cloud", np.random.default_rng(0).normal(size=(50, 3)), colors=[0.2, 0.6, 0.9], radii=0.05)
    log.scalars("sensor/T", [300.0, 301.0, 302.5], times=[0.0, 1.0, 2.0])
    g = ProofGraph("toy")
    g.assume("A", "a")
    g.add("L", "lemma", "1 = 1", ["A"], [symbolic_identity("id", 1, 1)])
    g.run()
    log.proof_graph(g)
    from pinneapple_physics.physics_audit import default_graph

    log.knowledge_graph(default_graph())
    log.close()
    assert path.exists() and path.stat().st_size > 1000


def test_plotly_and_trame(tmp_path):
    pytest.importorskip("plotly")
    from pinneapple_tools.visualization.integrations.plotly_export import (
        field_slices_html,
        scene_html,
    )

    p = scene_html(_scene(), str(tmp_path / "s.html"), field="p", include_plotlyjs="cdn")
    html = open(p).read()
    assert "mesh3d" in html and "Plotly.newPlot" in html
    x = np.linspace(-1, 1, 8)
    V = np.add.outer(np.add.outer(x ** 2, x ** 2), x ** 2)
    field_slices_html(x, x, x, V, str(tmp_path / "f.html"), iso=1.0, include_plotlyjs="cdn")
    assert "isosurface" in open(tmp_path / "f.html").read()
    pv = pytest.importorskip("pyvista")
    pytest.importorskip("trame_vtk")
    from pinneapple_tools.visualization.integrations.trame_app import build_app

    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(pv.Sphere())
    server = build_app(pl, title="test")
    assert server.state.title == "test"
    pl.close()
