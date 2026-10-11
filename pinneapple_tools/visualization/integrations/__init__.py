"""Bridges from PINNeAPPle results to other 3-D viewers.

* :mod:`.rerun_bridge`: Rerun (rerun.io), time-aware viewer for 3-D scenes, point clouds, time series, graphs and
  text on shared timelines, live or recorded to ``.rrd``;
* :mod:`.plotly_export`: self-contained interactive HTML (scenes, molecules, 3-D fields);
* :mod:`.trame_app`: Trame web app around a PyVista plotter (server-side or browser rendering).

The PINNeAPPle three.js viewer (:mod:`pinneapple_tools.visualization.studio`), PyVista, Blender Cycles and USD
export for Omniverse are in their own modules. Every bridge imports its library lazily.
"""
