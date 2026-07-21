#!/usr/bin/env python3
# ============================================================
# DERIVE PRECIPITATION FROM ANY AI-NWP MODEL'S ATMOSPHERIC FIELDS
#
# Statistically derives 6-hourly precipitation from an AI-NWP
# model's own forecast fields, using one of three pretrained
# models (OLS / LightGBM / MLP), all fitted on AIFS.
#
# To use with a NEW NWP model's output: edit the CONFIG block
# below only. See README.md for the exact variable/level
# requirements your forecast files must satisfy.
# ============================================================

import re
import pickle
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.ndimage import gaussian_filter
import metpy.calc as mpcalc
from metpy.units import units as munits

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG -- edit this block for your own NWP model / setup
# ============================================================
CONFIG = {
    "INPUT_DIR": "/path/to/your/forecast/files",
    "FILENAME_GLOB": "*_2024-*.nc",
    "FILENAME_REGEX": r"(\d{4}-\d{2}-\d{2})_(\d{2})-out-(\d+)",
    "FILENAME_TEMPLATE": "your_prefix_{date}_{hour}-out-{lead}.nc",
    "ENGINE": "netcdf4",
    "DOMAIN": {"lat_min": -12, "lat_max": 23, "lon_min": 92, "lon_max": 127},
    "MODEL_TYPE": "mlp",
    "MODEL_DIR": "/path/to/pretrained/models",
    "SMOOTHING_SIGMA": 1,
    "OUTPUT_DIR": "/path/to/output",
    "YEAR_FILTER": 2024,
}

# ============================================================
# Fixed model specification -- do not edit below this line
# unless you are retraining the models yourselves
# ============================================================
LOW_LEVEL   = 850
UPPER_LEVEL = 200
WIND_LEVEL  = 700
MID_LEVEL   = 500
PREDICTORS_9 = [
    "CONV_850", "DIV_200", "q_850", "T_850", "WSPD_700",
    "CONV850_x_q850", "TEND_CONV850_6h", "TEND_CONV850_12h", "TEND_q850_6h",
]
PREDICTORS_15 = PREDICTORS_9 + [
    "RH_700", "lapse_rate", "WSPD_850", "VORT_850", "WSPD_200", "DIV_500",
]
EPS = 0.1
VAR_MAP = {"u": ["u", "u10", "ua"], "v": ["v", "v10", "va"]}
SEASONS = {
    "NE": [11, 12, 1, 2, 3],
    "SW": [6, 7, 8, 9],
    "IM": [4, 5, 10],
}

INPUT_DIR      = Path(CONFIG["INPUT_DIR"])
OUTPUT_DIR     = Path(CONFIG["OUTPUT_DIR"])
MODEL_DIR      = Path(CONFIG["MODEL_DIR"])
DOMAIN         = CONFIG["DOMAIN"]
ENGINE         = CONFIG["ENGINE"]
MODEL_TYPE     = CONFIG["MODEL_TYPE"].lower()
PREDICTORS     = PREDICTORS_9 if MODEL_TYPE == "ols" else PREDICTORS_15
SMOOTHING_SIGMA = CONFIG["SMOOTHING_SIGMA"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if MODEL_TYPE == "mlp":
    import torch
    import torch.nn as nn

    class PrecipMLP(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 128),        nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64),         nn.ReLU(),
                nn.Linear(64, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_season(month):
    for name, months in SEASONS.items():
        if month in months:
            return name
    return "IM"

model_cache  = {}
scaler_cache = {}

def load_model(month):
    """Loads (and caches) the correct seasonal model for MODEL_TYPE."""
    season = get_season(month)
    if season in model_cache:
        return model_cache[season], scaler_cache.get(season)

    if MODEL_TYPE == "ols":
        df = pd.read_csv(MODEL_DIR / f"coefficients_{season}.csv").set_index("predictor")
        coeffs = df["coeff_train"]
        vec = np.array([coeffs["intercept"]] + [coeffs[p] for p in PREDICTORS], dtype=np.float64)
        model_cache[season] = vec
        print(f"Loaded OLS coefficients for {season}")
        return model_cache[season], None

    elif MODEL_TYPE == "lgbm":
        with open(MODEL_DIR / f"lgbm_model_{season}.pkl", "rb") as f:
            model_cache[season] = pickle.load(f)
        print(f"Loaded LightGBM model for {season}")
        return model_cache[season], None

    elif MODEL_TYPE == "mlp":
        model = PrecipMLP(len(PREDICTORS)).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_DIR / f"mlp_model_{season}.pt", map_location=DEVICE))
        model.eval()
        model_cache[season] = model
        with open(MODEL_DIR / f"mlp_scaler_{season}.pkl", "rb") as f:
            scaler_cache[season] = pickle.load(f)
        print(f"Loaded MLP model for {season}")
        return model_cache[season], scaler_cache[season]

    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE} (expected ols/lgbm/mlp)")

def predict_precipitation(X, model, scaler):
    """Runs prediction and inverse-transforms out of log-space, regardless of model type."""
    if MODEL_TYPE == "mlp":
        X_s = (X - scaler["mean"]) / scaler["std"]
        with torch.no_grad():
            ln_pred = model(torch.tensor(X_s, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    elif MODEL_TYPE == "ols":
        X_aug = np.column_stack([np.ones(len(X)), X])
        ln_pred = X_aug @ model
    else:
        ln_pred = model.predict(X)
    return np.clip(np.exp(ln_pred) - EPS, 0, None)

def parse_filename(fname):
    m = re.search(CONFIG["FILENAME_REGEX"], fname)
    if not m:
        raise ValueError(f"Filename did not match FILENAME_REGEX: {fname}")
    init_time = pd.Timestamp(f"{m.group(1)} {m.group(2)}:00")
    lead      = int(m.group(3))
    return {"init_time": init_time, "lead_time": lead,
            "valid_time": init_time + pd.Timedelta(hours=lead)}

def make_forecast_path(init_time, lead):
    """Reconstructs the filename for a different lead, same init_time,
    using FILENAME_TEMPLATE from CONFIG."""
    fname = CONFIG["FILENAME_TEMPLATE"].format(
        date=init_time.strftime('%Y-%m-%d'),
        hour=init_time.strftime('%H'),
        lead=lead,
    )
    return INPUT_DIR / fname

def standardize(ds):
    rename = {k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items() if k in ds.coords}
    return ds.rename(rename) if rename else ds

def normalize_lon(ds):
    if "lon" not in ds.coords:
        return ds
    if ds.lon.values.max() > 180:
        ds = ds.assign_coords(lon=(ds.lon.values + 180) % 360 - 180)
        ds = ds.sortby("lon")
    return ds

def clip_domain(ds):
    lats = ds["lat"].values
    if lats[0] > lats[-1]:
        return ds.sel(lat=slice(DOMAIN["lat_max"], DOMAIN["lat_min"]),
                      lon=slice(DOMAIN["lon_min"],  DOMAIN["lon_max"]))
    return ds.sel(lat=slice(DOMAIN["lat_min"], DOMAIN["lat_max"]),
                  lon=slice(DOMAIN["lon_min"], DOMAIN["lon_max"]))

def fix_time(ds, valid_time):
    if "time" in ds.dims:
        ds = ds.isel(time=0, drop=True)
    ds = ds.drop_vars([c for c in ("time", "step", "valid_time", "number")
                       if c in ds.coords], errors="ignore")
    return ds.expand_dims(time=[valid_time])

def open_forecast_file(fpath):
    """Opens a forecast file using whichever engine is configured.
    Works for GRIB, NetCDF, HDF5, Zarr, or anything else xarray supports."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Ignoring index file.*")
        kwargs = {}
        if ENGINE == "cfgrib":
            kwargs["backend_kwargs"] = {"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}}
        try:
            return xr.open_dataset(fpath, engine=ENGINE, **kwargs)
        except Exception:
            return xr.open_dataset(fpath, engine=ENGINE)

def sm(arr, lat, lon):
    dlat = abs(float(lat[1] - lat[0]))
    dlon = abs(float(lon[1] - lon[0]))
    smooth_deg = SMOOTHING_SIGMA * 0.25
    return gaussian_filter(arr, sigma=[smooth_deg / dlat, smooth_deg / dlon]).astype(np.float32)

def divergence_2d(u_arr, v_arr, lat, lon):
    u_da = xr.DataArray(u_arr.astype(float), coords={"lat": lat, "lon": lon}, dims=["lat", "lon"])
    v_da = xr.DataArray(v_arr.astype(float), coords={"lat": lat, "lon": lon}, dims=["lat", "lon"])
    div  = mpcalc.divergence(u_da.metpy.quantify(), v_da.metpy.quantify())
    return np.array(div.metpy.dequantify(), dtype=np.float32)

def vorticity_2d(u_arr, v_arr, lat, lon):
    R = 6371000.0
    dlat = np.deg2rad(np.abs(lat[1] - lat[0]))
    dlon = np.deg2rad(np.abs(lon[1] - lon[0]))
    dy = R * dlat
    lat_rad = np.deg2rad(lat)
    dx = R * np.cos(lat_rad) * dlon
    dvdx = np.gradient(v_arr, axis=1) / dx[:, None]
    dudy = np.gradient(u_arr, axis=0) / dy
    return (dvdx - dudy).astype(np.float32)

def sel_level(ds, var_name, level):
    da = ds[var_name]
    if "time" in da.dims:
        da = da.isel(time=0, drop=True)
    for dim in da.dims:
        if dim in ("lat", "lon", "latitude", "longitude"):
            continue
        da = da.sel({dim: level})
    return da.squeeze().values.astype(np.float32)

def get_uv_level(ds, level):
    def extract(aliases, component):
        for name in aliases:
            if name not in ds:
                continue
            da = ds[name]
            if "time" in da.dims:
                da = da.isel(time=0, drop=True)
            for dim in da.dims:
                if dim in ("lat", "lon", "latitude", "longitude"):
                    continue
                da = da.sel({dim: level})
                break
            return da.squeeze().values.astype(np.float32)
        raise KeyError(f"{component} not found -- check VAR_MAP matches your file's variable names")
    return extract(VAR_MAP["u"], "u"), extract(VAR_MAP["v"], "v")

def get_q850(ds):
    if "q" in ds:
        return sel_level(ds, "q", LOW_LEVEL)
    elif "r" in ds:
        t_arr = sel_level(ds, "t", LOW_LEVEL)
        r_arr = sel_level(ds, "r", LOW_LEVEL)
        if r_arr.max() > 1.5:
            r_arr = r_arr / 100.0
        t_K   = t_arr * munits.kelvin
        rh_fr = r_arr * munits.dimensionless
        w_s   = mpcalc.saturation_mixing_ratio(LOW_LEVEL * 100.0 * munits.pascal, t_K)
        q_arr = mpcalc.specific_humidity_from_mixing_ratio(rh_fr * w_s)
        return np.array(q_arr.to("kg/kg").magnitude, dtype=np.float32)
    raise KeyError("Neither q (specific humidity) nor r (relative humidity) found at 850hPa")

def get_rh700(ds):
    if "q" in ds:
        q700 = sel_level(ds, "q", WIND_LEVEL)
        t700 = sel_level(ds, "t", WIND_LEVEL)
        rh = mpcalc.relative_humidity_from_specific_humidity(
            WIND_LEVEL * 100 * munits.pascal,
            t700 * munits.kelvin,
            q700 * munits("kg/kg"),
        )
        return (rh.to("dimensionless").magnitude * 100).astype(np.float32)
    elif "r" in ds:
        r700 = sel_level(ds, "r", WIND_LEVEL)
        return r700 if r700.max() > 1.5 else r700 * 100.0
    return None

def compute_conv850_q850(fpath, lat, lon):
    ds = open_forecast_file(fpath)
    ds = standardize(ds); ds = normalize_lon(ds); ds = clip_domain(ds)
    if "time" in ds.dims:
        ds = ds.isel(time=0, drop=True)
    u850, v850 = get_uv_level(ds, LOW_LEVEL)
    conv850 = sm(-divergence_2d(u850, v850, lat, lon), lat, lon)
    q850    = get_q850(ds)
    ds.close()
    return conv850, q850

def derive_one(fpath, meta, model, scaler):
    ds  = open_forecast_file(fpath)
    ds  = standardize(ds); ds = normalize_lon(ds); ds = clip_domain(ds)
    ds  = fix_time(ds, meta["valid_time"])

    lat = ds.lat.values
    lon = ds.lon.values

    u850, v850 = get_uv_level(ds, LOW_LEVEL)
    conv850    = sm(-divergence_2d(u850, v850, lat, lon), lat, lon)
    wspd850    = sm(np.sqrt(u850**2 + v850**2).astype(np.float32), lat, lon)
    vort850    = sm(vorticity_2d(u850, v850, lat, lon), lat, lon)

    u200, v200 = get_uv_level(ds, UPPER_LEVEL)
    div200     = sm(divergence_2d(u200, v200, lat, lon), lat, lon)
    wspd200    = sm(np.sqrt(u200**2 + v200**2).astype(np.float32), lat, lon)

    try:
        u700, v700 = get_uv_level(ds, WIND_LEVEL)
        wspd700    = sm(np.sqrt(u700**2 + v700**2).astype(np.float32), lat, lon)
    except Exception:
        wspd700 = np.zeros_like(conv850)

    try:
        u500, v500 = get_uv_level(ds, MID_LEVEL)
        div500     = sm(divergence_2d(u500, v500, lat, lon), lat, lon)
    except Exception:
        div500 = np.zeros_like(conv850)

    t850 = sel_level(ds, "t", LOW_LEVEL)
    t500 = sel_level(ds, "t", MID_LEVEL)
    q850 = get_q850(ds)
    lapse_rate = t850 - t500

    rh700 = get_rh700(ds)
    if rh700 is None:
        rh700 = np.zeros_like(conv850)

    ds.close()

    lead      = meta["lead_time"]
    init_time = meta["init_time"]

    path_tm6 = make_forecast_path(init_time, lead - 6)
    if lead >= 6 and path_tm6.exists():
        try:
            conv850_tm6, q850_tm6 = compute_conv850_q850(path_tm6, lat, lon)
            tend_conv850_6h = conv850 - conv850_tm6
            tend_q850_6h    = q850   - q850_tm6
        except Exception:
            tend_conv850_6h = np.zeros_like(conv850)
            tend_q850_6h    = np.zeros_like(q850)
    else:
        tend_conv850_6h = np.zeros_like(conv850)
        tend_q850_6h    = np.zeros_like(q850)

    path_tm12 = make_forecast_path(init_time, lead - 12)
    if lead >= 12 and path_tm12.exists():
        try:
            conv850_tm12, _ = compute_conv850_q850(path_tm12, lat, lon)
            tend_conv850_12h = conv850 - conv850_tm12
        except Exception:
            tend_conv850_12h = np.zeros_like(conv850)
    else:
        tend_conv850_12h = np.zeros_like(conv850)

    preds = {
        "CONV_850":          conv850,
        "DIV_200":           div200,
        "q_850":             q850,
        "T_850":             t850,
        "WSPD_700":          wspd700,
        "CONV850_x_q850":    conv850 * q850,
        "TEND_CONV850_6h":   tend_conv850_6h,
        "TEND_CONV850_12h":  tend_conv850_12h,
        "TEND_q850_6h":      tend_q850_6h,
        "RH_700":            rh700,
        "lapse_rate":        lapse_rate,
        "WSPD_850":          wspd850,
        "VORT_850":          vort850,
        "WSPD_200":          wspd200,
        "DIV_500":           div500,
    }

    X = np.column_stack([preds[p].ravel().astype(np.float32) for p in PREDICTORS])
    precip = predict_precipitation(X, model, scaler)
    precip = precip.reshape(len(lat), len(lon)).astype(np.float32)
    return lat, lon, precip

if __name__ == "__main__":
    all_files = sorted(INPUT_DIR.glob(CONFIG["FILENAME_GLOB"]))
    files = []
    for f in all_files:
        try:
            meta = parse_filename(f.name)
            if CONFIG["YEAR_FILTER"] is not None and meta["init_time"].year != CONFIG["YEAR_FILTER"]:
                continue
            if meta["lead_time"] <= 0:
                continue
            files.append((f, meta))
        except Exception:
            continue

    print(f"Model type : {MODEL_TYPE.upper()}")
    print(f"Files      : {len(files)}")
    print(f"Output     : {OUTPUT_DIR}\n")

    ok = skipped = 0
    for f, meta in files:
        out_path = OUTPUT_DIR / (
            f"derived_{meta['init_time'].strftime('%Y-%m-%d_%H')}"
            f"-out-{meta['lead_time']}.nc"
        )
        if out_path.exists():
            skipped += 1
            continue
        print(f"  {f.name} ...", end=" ", flush=True)
        try:
            model, scaler = load_model(meta["init_time"].month)
            lat, lon, precip = derive_one(f, meta, model, scaler)
            ds_out = xr.Dataset({"precip_derived": (["lat", "lon"], precip)},
                                coords={"lat": lat, "lon": lon})
            ds_out["precip_derived"].attrs = {"units": "mm", "method": MODEL_TYPE.upper()}
            ds_out.attrs = {"model_type": MODEL_TYPE,
                           "init_time": str(meta["init_time"]),
                           "valid_time": str(meta["valid_time"]),
                           "lead_time": meta["lead_time"]}
            ds_out.to_netcdf(out_path)
            print("OK")
            ok += 1
        except Exception as e:
            print(f"FAIL — {e}")
            skipped += 1

    print(f"\nOK: {ok}  Skipped: {skipped}")
