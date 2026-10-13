"""Library entries: every module in this package registers its geometries on import (discovered automatically)."""
import importlib
import pkgutil

for _m in sorted(m.name for m in pkgutil.iter_modules(__path__) if not m.name.startswith("_")):
    importlib.import_module(f"{__name__}.{_m}")
