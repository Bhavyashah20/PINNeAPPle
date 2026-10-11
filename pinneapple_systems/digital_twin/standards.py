"""Digital-twin interoperability: describe a PINNeAPPle twin once and publish it in the open standards industry
platforms read.

=============================  ==========================================================================
standard                       what it is, and the function here
=============================  ==========================================================================
Asset Administration Shell     Industrie 4.0 / IDTA (IEC 63278): the asset's shell with submodels (identification,
(AAS, metamodel v3)            technical data, operational data, simulation models). :func:`to_aas` builds it
                               with the Eclipse BaSyx Python SDK and returns its JSON
DTDL v3                        Digital Twins Definition Language (Azure Digital Twins, also read by other
                               platforms): an Interface with Properties, Telemetry and Relationships. :func:`to_dtdl`
NGSI-LD                        ETSI context-information standard (FIWARE Orion-LD and others): an entity with
                               Property / Relationship attributes. :func:`to_ngsi_ld`
Eclipse Ditto                  open-source twin service: a Thing with attributes and features. :func:`to_ditto_thing`
OPC UA                         IEC 62541, the shop-floor protocol: :class:`OpcUaTwinServer` serves the twin's
                               values live (asyncua) so SCADA, MES and OPC UA clients read and subscribe to them
=============================  ==========================================================================

A :class:`TwinDescription` holds the asset (id, name, type), its parameters, live telemetry with units, the models
behind the twin and their physical validity (from :mod:`pinneapple_physics.physics_audit`: assumption status,
compatibility, the probable failure cause), so the trust layer travels with the twin into every standard.
"""
from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["TwinProperty", "TwinModel", "TwinDescription", "to_aas", "to_dtdl", "to_ngsi_ld", "to_ditto_thing",
           "OpcUaTwinServer"]

_KINDS = ("parameter", "telemetry", "state")


@dataclass
class TwinProperty:
    name: str
    value: float | int | str | bool
    unit: str = ""
    kind: str = "parameter"          # parameter (design / configuration), telemetry (measured), state (estimated)
    description: str = ""
    writable: bool = False

    def __post_init__(self):
        if self.kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", self.name):
            raise ValueError(f"property name {self.name!r}: letters, digits and _ only, starting with a letter")


@dataclass
class TwinModel:
    """A model behind the twin and its physical validity on the current operating point."""
    name: str
    family: str = ""
    status: str = ""                 # VALID / MARGINAL / VIOLATED / UNKNOWN (physics audit)
    compatibility: float | None = None
    violated: list[str] = field(default_factory=list)
    probable_cause: str = ""

    @classmethod
    def from_audit(cls, audit, family: str = "", attribution=None) -> TwinModel:
        return cls(audit.model, family, audit.status, float(audit.compatibility), [c.name for c in audit.violated],
                   attribution.top.title if attribution is not None and attribution.failed else "")


@dataclass
class TwinDescription:
    id: str                          # globally unique, e.g. "urn:pinneapple:twin:pump-17"
    name: str
    asset_type: str = "Asset"
    description: str = ""
    manufacturer: str = ""
    properties: list[TwinProperty] = field(default_factory=list)
    models: list[TwinModel] = field(default_factory=list)
    relationships: list[tuple[str, str]] = field(default_factory=list)   # (name, target twin id)

    @classmethod
    def from_snapshot(cls, id: str, name: str, snapshot: Mapping[str, Any], units: Mapping[str, str] | None = None,
                      **kw) -> TwinDescription:
        """From ``DigitalTwin.snapshot()`` (or any {field: value}): scalar fields become telemetry, arrays their
        mean (the full field stays in the twin)."""
        import numpy as np

        props = []
        fields = snapshot.get("fields", snapshot)
        for k, v in fields.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                props.append(TwinProperty(k, float(v), (units or {}).get(k, ""), "telemetry"))
            elif hasattr(v, "__len__") and not isinstance(v, str):
                a = np.asarray(v, dtype=float)
                if a.size:
                    props.append(TwinProperty(f"{k}_mean", float(np.nanmean(a)), (units or {}).get(k, ""), "state",
                                              f"mean of the estimated field {k}"))
        return cls(id, name, properties=props, **kw)

    def by_kind(self, kind: str) -> list[TwinProperty]:
        return [p for p in self.properties if p.kind == kind]


# ------------------------------------------------------------------------------------------------ AAS
def to_aas(desc: TwinDescription, as_json: bool = True):
    """Asset Administration Shell (metamodel v3) with submodels Identification, TechnicalData (parameters),
    OperationalData (telemetry and estimated state) and SimulationModels (models and physical validity), built
    with the Eclipse BaSyx Python SDK. Returns the JSON text (``as_json``) or the BaSyx object store."""
    from basyx.aas import model
    from basyx.aas.adapter import json as aasjson

    def txt(s: str):
        return model.MultiLanguageTextType({"en": s[:1023]}) if s else None

    def prop(id_short, value, unit="", description=""):
        vt = bool if isinstance(value, bool) else float if isinstance(value, float) else int if isinstance(value, int) \
            else str
        d = (description + (f" [{unit}]" if unit else "")).strip()
        return model.Property(id_short=id_short, value_type=vt, value=value, description=txt(d))

    base = desc.id.rstrip("/")
    sm_ident = model.Submodel(f"{base}/sm/identification", id_short="Identification", submodel_element=[
        prop("Name", desc.name), prop("AssetType", desc.asset_type),
        *([prop("Manufacturer", desc.manufacturer)] if desc.manufacturer else []),
        *([prop("Description", desc.description)] if desc.description else [])])
    sm_tech = model.Submodel(f"{base}/sm/technical-data", id_short="TechnicalData", submodel_element=[
        prop(p.name, p.value, p.unit, p.description) for p in desc.by_kind("parameter")])
    sm_op = model.Submodel(f"{base}/sm/operational-data", id_short="OperationalData", submodel_element=[
        prop(p.name, p.value, p.unit, (p.description + f" ({p.kind})").strip()) for p in
        desc.by_kind("telemetry") + desc.by_kind("state")])
    models = []
    for m in desc.models:
        els = [prop("Family", m.family or "unknown"), prop("ValidityStatus", m.status or "UNKNOWN")]
        if m.compatibility is not None:
            els.append(prop("Compatibility", float(m.compatibility), "", "physics-audit compatibility, 0 to 1"))
        if m.violated:
            els.append(prop("ViolatedAssumptions", ", ".join(m.violated)))
        if m.probable_cause:
            els.append(prop("ProbableFailureCause", m.probable_cause))
        models.append(model.SubmodelElementCollection(id_short=re.sub(r"[^A-Za-z0-9_]", "_", m.name)[:128] or "model",
                                                      value=els))
    sm_models = model.Submodel(f"{base}/sm/simulation-models", id_short="SimulationModels", submodel_element=models)
    submodels = [sm_ident, sm_tech, sm_op, sm_models]
    shell = model.AssetAdministrationShell(
        asset_information=model.AssetInformation(global_asset_id=f"{base}/asset", asset_type=desc.asset_type),
        id_=f"{base}/aas", id_short=re.sub(r"[^A-Za-z0-9_]", "_", desc.name)[:128] or "Twin",
        description=txt(desc.description),
        submodel={model.ModelReference.from_referable(s) for s in submodels})
    store = model.DictObjectStore([shell, *submodels])
    if not as_json:
        return store
    return aasjson.object_store_to_json(store)


# ------------------------------------------------------------------------------------------------ DTDL
def _dtmi(desc: TwinDescription, version: int = 1) -> str:
    seg = [re.sub(r"[^A-Za-z0-9_]", "_", s) for s in re.split(r"[:/.]+", desc.id) if s and s.lower() != "urn"]
    seg = [s if s[0].isalpha() else "x" + s for s in seg if s]
    return "dtmi:" + ":".join(seg or ["pinneapple", "twin"]) + f";{version}"


def to_dtdl(desc: TwinDescription, version: int = 1) -> dict[str, Any]:
    """A DTDL v3 Interface: parameters and estimated state as Properties, measurements as Telemetry, model validity
    as Properties, links to other twins as Relationships."""
    def schema(v):
        return "boolean" if isinstance(v, bool) else "double" if isinstance(v, float) else \
            "integer" if isinstance(v, int) else "string"

    contents: list[dict[str, Any]] = []
    for p in desc.properties:
        c = {"@type": "Telemetry" if p.kind == "telemetry" else "Property", "name": p.name, "schema": schema(p.value)}
        d = (p.description + (f" [{p.unit}]" if p.unit else "")).strip()
        if d:
            c["description"] = d[:512]
        if p.kind != "telemetry":
            c["writable"] = bool(p.writable)
        contents.append(c)
    for m in desc.models:
        nm = re.sub(r"[^A-Za-z0-9_]", "_", m.name)
        contents.append({"@type": "Property", "name": f"validity_{nm}"[:64], "schema": "string",
                         "description": f"physics-audit status of model {m.name} ({m.family})"})
    for rel, _ in desc.relationships:
        contents.append({"@type": "Relationship", "name": re.sub(r"[^A-Za-z0-9_]", "_", rel)[:64]})
    out = {"@context": "dtmi:dtdl:context;3", "@id": _dtmi(desc, version), "@type": "Interface",
           "displayName": desc.name[:64], "contents": contents}
    if desc.description:
        out["description"] = desc.description[:512]
    return out


# ------------------------------------------------------------------------------------------------ NGSI-LD / Ditto
def to_ngsi_ld(desc: TwinDescription, context: str = "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld",
               observed_at: str | None = None) -> dict[str, Any]:
    """An NGSI-LD entity: every property as a ``Property`` (unit as ``unitText``, measurements with
    ``observedAt``), model validity as properties, links as ``Relationship``s."""
    eid = desc.id if desc.id.startswith("urn:ngsi-ld:") else "urn:ngsi-ld:" + re.sub(r"^urn:", "", desc.id)
    ent: dict[str, Any] = {"id": eid, "type": desc.asset_type, "name": {"type": "Property", "value": desc.name},
                           "@context": context}
    for p in desc.properties:
        a: dict[str, Any] = {"type": "Property", "value": p.value}
        if p.unit:
            a["unitText"] = p.unit
        if p.kind == "telemetry" and observed_at:
            a["observedAt"] = observed_at
        ent[p.name] = a
    for m in desc.models:
        ent[f"validity_{re.sub(r'[^A-Za-z0-9_]', '_', m.name)}"] = {
            "type": "Property", "value": m.status or "UNKNOWN",
            **({"compatibility": {"type": "Property", "value": m.compatibility}} if m.compatibility is not None else {})}
    for rel, target in desc.relationships:
        ent[rel] = {"type": "Relationship", "object": target}
    return ent


def to_ditto_thing(desc: TwinDescription, namespace: str = "org.pinneapple") -> dict[str, Any]:
    """An Eclipse Ditto Thing: identification and parameters as attributes; telemetry, state and model validity as
    features."""
    name = re.sub(r"[^A-Za-z0-9_.-]", "-", desc.name)
    feats: dict[str, Any] = {}
    for kind in ("telemetry", "state"):
        props = {p.name: {"value": p.value, **({"unit": p.unit} if p.unit else {})} for p in desc.by_kind(kind)}
        if props:
            feats[kind] = {"properties": props}
    if desc.models:
        feats["models"] = {"properties": {m.name: {"family": m.family, "status": m.status,
                                                    "compatibility": m.compatibility, "violated": m.violated,
                                                    "probableCause": m.probable_cause} for m in desc.models}}
    attrs = {"name": desc.name, "assetType": desc.asset_type, "id": desc.id,
             **({"manufacturer": desc.manufacturer} if desc.manufacturer else {}),
             "parameters": {p.name: {"value": p.value, **({"unit": p.unit} if p.unit else {})}
                            for p in desc.by_kind("parameter")},
             "relationships": {r: t for r, t in desc.relationships}}
    return {"thingId": f"{namespace}:{name}", "attributes": attrs, "features": feats}


# ------------------------------------------------------------------------------------------------ OPC UA
class OpcUaTwinServer:
    """Serve a twin over OPC UA (asyncua) in a background thread: one object per twin with a variable per property
    (and the validity status of each model). :meth:`update` writes new values; OPC UA clients read or subscribe.

        srv = OpcUaTwinServer(desc, "opc.tcp://0.0.0.0:4840/pinneapple/").start()
        srv.update("outlet_temperature", 351.2)
        ...
        srv.stop()

    No security policy is configured (anonymous, unencrypted): fine on a lab network, configure certificates for
    production."""

    def __init__(self, desc: TwinDescription, endpoint: str = "opc.tcp://127.0.0.1:4840/pinneapple/",
                 namespace: str = "urn:pinneapple:twins"):
        self.desc, self.endpoint, self.namespace = desc, endpoint, namespace
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self.nodes: dict[str, Any] = {}
        self.node_ids: dict[str, str] = {}
        self.error: BaseException | None = None

    async def _main(self):
        from asyncua import Server, ua

        server = Server()
        await server.init()
        server.set_endpoint(self.endpoint)
        server.set_server_name(f"PINNeAPPle twin {self.desc.name}")
        idx = await server.register_namespace(self.namespace)
        obj = await server.nodes.objects.add_object(idx, re.sub(r"[^A-Za-z0-9_]", "_", self.desc.name))
        for p in self.desc.properties:
            v = await obj.add_variable(idx, p.name, p.value)
            if p.writable:
                await v.set_writable()
            if p.unit:
                await v.add_property(idx, f"{p.name}_EngineeringUnits", p.unit)
            self.nodes[p.name] = v
            self.node_ids[p.name] = v.nodeid.to_string()
        for m in self.desc.models:
            nm = f"validity_{re.sub(r'[^A-Za-z0-9_]', '_', m.name)}"
            v = await obj.add_variable(idx, nm, m.status or "UNKNOWN")
            self.nodes[nm] = v
            self.node_ids[nm] = v.nodeid.to_string()
        self._stop = asyncio.Event()
        async with server:
            self._ready.set()
            await self._stop.wait()
        _ = ua

    def _run(self):
        self._loop = asyncio.new_event_loop()
        try:
            self._loop.run_until_complete(self._main())
        except BaseException as exc:  # noqa: BLE001 - surfaced to the caller of start()
            self.error = exc
            self._ready.set()
        finally:
            self._loop.close()

    def start(self, timeout: float = 20.0) -> OpcUaTwinServer:
        self._thread = threading.Thread(target=self._run, daemon=True, name="opcua-twin")
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("OPC UA server did not start")
        if self.error:
            raise RuntimeError(f"OPC UA server failed: {self.error}") from self.error
        return self

    def update(self, name: str, value) -> None:
        if name not in self.nodes:
            raise KeyError(f"unknown variable {name!r}")
        fut = asyncio.run_coroutine_threadsafe(self.nodes[name].write_value(value), self._loop)
        fut.result(timeout=10)

    def update_many(self, values: Mapping[str, Any] | Sequence[tuple[str, Any]]) -> None:
        for k, v in (values.items() if isinstance(values, Mapping) else values):
            self.update(k, v)

    def stop(self) -> None:
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(timeout=15)
