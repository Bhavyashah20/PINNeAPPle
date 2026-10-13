"""
NASA POWER (Daily) -> UPD (PhysicalSample) -> Zarr
(+ optional pinneapple_pdb ingestion if available)

Ref:
  /api/temporal/daily/point?parameters=...&community=...&longitude=...&latitude=...&start=...&end=...&format=JSON
  (NASA POWER Daily API docs)
"""

from __future__ import annotations

import json
import os

from pinneapple_data.datasets.external_sources import NASAPowerClient
from pinneapple_data.zarr_store import UPDZarrStore


def main():
    out_dir = "examples/_out/pdb_nasa_power"
    os.makedirs(out_dir, exist_ok=True)

    zarr_path = os.path.join(out_dir, "nasa_power_daily_point.zarr")

    # Example: Recife-ish lat/lon (adjust as you want)
    latitude = -8.05
    longitude = -34.9

    # Choose a small window for demo
    start = "20240101"
    end = "20240131"

    parameters = ["T2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"]

    client = NASAPowerClient()
    power = client.fetch_daily(
        latitude=latitude,
        longitude=longitude,
        start=start,
        end=end,
        parameters=parameters,
        community="SB",
        time_standard="UTC",
    )

    samples = client.to_physical_samples(power)

    # Write UPD -> Zarr
    UPDZarrStore.write(
        zarr_path,
        samples,
        manifest={
            "name": "nasa_power_daily_point",
            "source": "NASA POWER Daily API",
            "latitude": latitude,
            "longitude": longitude,
            "start": start,
            "end": end,
            "parameters": parameters,
        },
    )

    print("wrote:", zarr_path)
    
    # save raw for inspection
    raw_path = os.path.join(out_dir, "raw_power.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(power, f, indent=2)
    print("saved raw:", raw_path)


if __name__ == "__main__":
    main()
