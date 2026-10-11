# Other 3-D viewers and digital-twin standards

PINNeAPPle already renders with its own three.js viewer, PyVista, Blender Cycles, and USD for Omniverse. The bridges
below add three more viewers, and the digital-twin exporters publish a twin in the formats industrial platforms read.
Each library is imported only when used.

## 3-D viewers (`pinneapple_tools.visualization.integrations`)

| Bridge | Library | Good for |
|---|---|---|
| `rerun_bridge.RerunLogger` | [Rerun](https://rerun.io) (`rerun-sdk`) | Time-aware viewing: 3-D scenes, point clouds, molecules frame by frame, time series (sensors, errors, ensemble weights), graphs (proof graph, knowledge graph) and Markdown reports, on shared timelines, live or saved as `.rrd` |
| `plotly_export` | Plotly | A self-contained interactive HTML file (scene with colour bar, molecule with hover, 3-D field with slices and isosurface) that opens in any browser |
| `trame_app.build_app` | Trame (Kitware) | A web app around a PyVista plotter, rendered on the server or in the browser, for meshes too large for a standalone file |

```python
from pinneapple_tools.visualization.integrations.rerun_bridge import RerunLogger
log = RerunLogger("pump-17", path="pump.rrd")      # or spawn=True for a live viewer
log.scene(scene, field="p")
log.ensemble_run(run)                              # weights, errors, active model per case
log.proof_graph(graph)                             # nodes coloured by status, with the report
log.close()                                        # then: rerun pump.rrd
```

## Digital-twin standards (`pinneapple_systems.digital_twin.standards`)

A `TwinDescription` holds the asset, its parameters, live telemetry and estimated state with units, the models
behind the twin and their physical validity. The validity comes from `physics_audit`: status, compatibility,
violated assumptions and probable failure cause. That trust information travels into every format.

| Function | Standard | Output |
|---|---|---|
| `to_aas` | Asset Administration Shell, metamodel v3 (Industrie 4.0 / IDTA, IEC 63278), built with Eclipse BaSyx | A shell plus the submodels Identification, TechnicalData, OperationalData and SimulationModels, as JSON |
| `to_dtdl` | DTDL v3 (Azure Digital Twins) | An Interface with Properties, Telemetry and Relationships |
| `to_ngsi_ld` | NGSI-LD (ETSI, FIWARE) | An entity with Property and Relationship attributes, units and `observedAt` |
| `to_ditto_thing` | Eclipse Ditto | A Thing with attributes and features |
| `OpcUaTwinServer` | OPC UA (IEC 62541), via asyncua | A live server with one variable per property and the validity of each model; `update()` writes new values |

The OPC UA server runs without a security policy (anonymous, unencrypted). Configure certificates before using it
outside a lab network.

Install the extras with `pip install pinneapple[viz3d]` (Rerun, Plotly, Trame) and `pip install pinneapple[twin]`
(BaSyx, asyncua). Tests: `tests/test_viz_twin_integrations.py`. Among the tests, the AAS export is read back by
BaSyx and the OPC UA values are read by an OPC UA client.
