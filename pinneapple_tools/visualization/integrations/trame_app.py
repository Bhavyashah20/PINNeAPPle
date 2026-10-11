"""Trame (Kitware) web app around a PyVista plotter: the VTK scene rendered on the server or in the browser, with a
small control panel, served from Python (no JavaScript to write). Good for heavy meshes and fields that are too big
for a standalone HTML file, inside a digital-twin dashboard or on a workstation shared over the network.

    from pinneapple_tools.visualization.integrations.trame_app import build_app
    server = build_app(plotter, title="Ahmed body: pressure")
    server.start(port=8080)                                   # http://localhost:8080

Needs ``trame``, ``trame-vtk`` and ``trame-vuetify`` (plus ``pyvista``).
"""
from __future__ import annotations

__all__ = ["build_app"]


def build_app(plotter, title: str = "PINNeAPPle", local_rendering: bool = False, server_name: str | None = None):
    """A trame server showing ``plotter``'s render window (``local_rendering``: vtk.js in the browser instead of
    images streamed from the server). Returns the server; call ``server.start()`` to serve it."""
    from trame.app import get_server
    from trame.ui.vuetify import SinglePageLayout
    from trame.widgets import vtk as vtk_widgets
    from trame.widgets import vuetify

    server = get_server(server_name or f"pinneapple-{id(plotter)}", client_type="vue2")
    ren_win = plotter.ren_win
    with SinglePageLayout(server) as layout:
        layout.title.set_text(title)
        with layout.toolbar:
            vuetify.VSpacer()
            vuetify.VBtn("Reset camera", click="$refs.view.resetCamera()", small=True)
        with layout.content:
            with vuetify.VContainer(fluid=True, classes="pa-0 fill-height"):
                View = vtk_widgets.VtkLocalView if local_rendering else vtk_widgets.VtkRemoteView
                View(ren_win, ref="view")
    server.state.title = title
    return server
