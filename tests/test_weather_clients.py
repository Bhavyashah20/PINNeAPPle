"""NASAPowerClient and OpenMeteoClient with a mocked HTTP response (no network)."""
import io
import json
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pytest

from pinneapple_data.datasets.external_sources import NASAPowerClient, OpenMeteoClient

POWER_JSON = {
    "header": {"title": "NASA/POWER Source Native Resolution Daily Data", "api": {"version": "v2"},
               "fill_value": -999.0, "start": "20240101", "end": "20240103"},
    "parameters": {"T2M": {"units": "C"}, "PRECTOTCORR": {"units": "mm/day"}},
    "properties": {"parameter": {
        "T2M": {"20240101": 27.1, "20240102": -999.0, "20240103": 26.4},
        "PRECTOTCORR": {"20240101": 0.0, "20240102": 3.5, "20240103": None},
    }},
}

METEO_JSON = {
    "latitude": -5.8, "longitude": -35.2, "timezone": "America/Fortaleza",
    "hourly": {"time": ["2026-08-01T00:00", "2026-08-01T01:00", "2026-08-01T02:00"],
               "temperature_2m": [25.1, 24.8, None], "shortwave_radiation": [0.0, 0.0, 12.0]},
}


class _Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


@pytest.fixture
def fake_http(monkeypatch):
    calls = []

    def install(payload):
        def urlopen(req, timeout=None):
            calls.append((req.full_url, timeout))
            return _Resp(json.dumps(payload).encode())
        monkeypatch.setattr(urllib.request, "urlopen", urlopen)
        return calls
    return install


def _query(url):
    return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))


def test_nasa_power_request_and_arrays(fake_http):
    calls = fake_http(POWER_JSON)
    c = NASAPowerClient()
    js = c.fetch_daily(latitude=-8.05, longitude=-34.9, start="20240101", end="20240103",
                       parameters=["T2M", "PRECTOTCORR"])
    url, timeout = calls[0]
    assert url.startswith(NASAPowerClient.DAILY_URL) and timeout == 60
    assert _query(url) == {"parameters": "T2M,PRECTOTCORR", "community": "SB", "longitude": "-34.9",
                           "latitude": "-8.05", "start": "20240101", "end": "20240103", "format": "JSON",
                           "time-standard": "UTC"}
    arr = c.to_arrays(js)
    assert list(arr) == ["date", "PRECTOTCORR", "T2M"]
    assert arr["date"].tolist() == np.array(["2024-01-01", "2024-01-02", "2024-01-03"], "datetime64[D]").tolist()
    np.testing.assert_array_equal(arr["T2M"], [27.1, np.nan, 26.4])            # fill value -> NaN
    np.testing.assert_array_equal(arr["PRECTOTCORR"], [0.0, 3.5, np.nan])      # null -> NaN


def test_nasa_power_physical_samples():
    pytest.importorskip("xarray")
    (s,) = NASAPowerClient.to_physical_samples(POWER_JSON)
    assert tuple(s.state["x"].shape) == (3, 2) and tuple(s.state["t"].shape) == (3, 1)
    assert s.schema["columns"] == ["PRECTOTCORR", "T2M"] and s.schema["units"]["T2M"] == "C"
    assert s.provenance["source"] == "nasa_power"


def test_nasa_power_rejects_bad_payload():
    with pytest.raises(ValueError):
        NASAPowerClient.to_arrays({"properties": {}})


def test_open_meteo_request_and_arrays(fake_http):
    calls = fake_http(METEO_JSON)
    d = OpenMeteoClient().fetch_archive(-5.795, -35.209, "2026-08-01", "2026-08-01",
                                        ["temperature_2m", "shortwave_radiation"], timezone="America/Fortaleza")
    q = _query(calls[0][0])
    assert calls[0][0].startswith(OpenMeteoClient.ARCHIVE_URL)
    assert q["hourly"] == "temperature_2m,shortwave_radiation" and q["timezone"] == "America/Fortaleza"
    assert q["start_date"] == q["end_date"] == "2026-08-01" and q["latitude"] == "-5.795"
    assert d["time"].dtype == np.dtype("datetime64[m]") and d["time"][1] == np.datetime64("2026-08-01T01:00")
    np.testing.assert_array_equal(d["temperature_2m"], [25.1, 24.8, np.nan])
    np.testing.assert_array_equal(d["shortwave_radiation"], [0.0, 0.0, 12.0])


def test_open_meteo_http_error_raises(monkeypatch):
    def urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", None, None)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with pytest.raises(urllib.error.HTTPError):
        OpenMeteoClient().fetch_archive_json(0.0, 0.0, "2026-01-01", "2026-01-02", ["temperature_2m"])
