# Environment SimVars

The Infinity Manager reads these and pushes them into `infinity_viewports_protocol::Environment
Wire` (consumed by `flir-env`). This crate does **no** SimConnect itself; the
list is the contract so the Manager knows exactly what to read. Names must be
verified against the MSFS 2024 SDK SimVar reference for your SU; those marked
*(verify)* have changed names/availability across versions in the past.

| `EnvironmentWire` field | SimVar | Unit | Notes |
|---|---|---|---|
| `ambient_temp_c` | `AMBIENT TEMPERATURE` | celsius | At the aircraft. |
| `pressure_hpa` | `AMBIENT PRESSURE` | millibars (= hPa) | Or `SEA LEVEL PRESSURE` if you prefer QNH; pick one and be consistent. |
| `humidity_pct` | `AMBIENT DENSITY` → derived, or none | — | MSFS exposes no direct RH SimVar. Leave `NaN` and let `flir-env` estimate, **or** compute RH from temperature + dew point if you source one. |
| `dewpoint_c` | *(none native)* | celsius | No reliable SimVar; set `NaN` unless sourced from METAR. |
| `visibility_m` | `AMBIENT VISIBILITY` | meters | Drives the humidity fallback. |
| `precip_kind` | `AMBIENT PRECIP STATE` | enum → 0/1/2 | Bitmask in-sim; map rain→1, snow→2, else 0. *(verify)* |
| `precip_rate` | `AMBIENT PRECIP RATE` | mm/h *(verify unit)* | 0 when none. |
| `in_cloud` | `AMBIENT IN CLOUD` | bool | 1 when the aircraft is in cloud. *(verify)* |
| `wind_dir_deg` | `AMBIENT WIND DIRECTION` | degrees (true) | |
| `wind_speed_ms` | `AMBIENT WIND VELOCITY` | knots → m/s | Convert (×0.514444). |
| `lat_deg` | `PLANE LATITUDE` | degrees | |
| `lon_deg` | `PLANE LONGITUDE` | degrees | |
| `alt_m` | `PLANE ALTITUDE` | feet → m | Geometric; convert (×0.3048). |
| `ground_alt_m` | `GROUND ALTITUDE` | meters | Already metres in the SDK. *(verify)* |
| `utc_unix_s` | `ZULU TIME` + `ZULU YEAR`/`ZULU DAY OF YEAR` | → Unix seconds | Compose a UTC timestamp; used by the NOAA solar model. Alternatively feed real wall-clock UTC if the sim clock tracks it. |
| `sun_dir` | *(engine, optional)* | unit vec | Leave all-`NaN` to use NOAA. If a future build exposes a sun vector in the view constant buffer, the bridge can fill this instead. |

## Humidity when not available

MSFS has no direct relative-humidity SimVar. `flir-env::atmos::humidity_from_
visibility` estimates RH from `AMBIENT VISIBILITY` and precipitation (documented
heuristic: low visibility / active precip ⇒ near-saturation). If the Manager has
METAR dew point, prefer `humidity_from_dewpoint`, which is physical. The
`Environment::humidity_estimated` flag records which path was taken, and the
bridge reports it so the look can be judged accordingly.

## Push rate

1–5 Hz is plenty; the thermal response is slow. The scene model reads the latest
seqlock snapshot each frame, so a stale environment simply persists — never
blocks a frame.
