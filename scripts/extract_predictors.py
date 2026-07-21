#!/usr/bin/env python3
import re
import gc
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from scipy.ndimage import gaussian_filter
import metpy.calc as mpcalc
from metpy.units import units as munits

warnings.filterwarnings("ignore")

CONFIG = {
    "INPUT_DIR": "/path/to/aifs/forecasts",
    "FILENAME_REGEX": r"(\d{4}-\d{2}-\d{2})_(\d{2})-out-(\d+)",
    "FILENAME_TEMPLATE": "aifs_gridded_{date}_{hour}-out-{lead}_regridded.nc",
    "DOMAIN": {"lat_min": -12, "lat_max": 23, "lon_min": 92, "lon_max": 127},
    "SMOOTHING_SIGMA": 10,
    "SUBSAMPLE_STRIDE": 4,
    "RAIN_MM_THRESHOLD": 0.1,
    "OUTPUT_PARQUET": "/path/to/output/precip_predictors.parquet",
}

LOW_LEVEL   = 850
UPPER_LEVEL = 200
WIND_LEVEL  = 700
MID_LEVEL   = 500
EPS         = 0.1
VAR_MAP = {
    "rain": ["tp", "precip", "total_precipitation"],
    "u":    ["u", "u10", "ua"],
    "v":    ["v", "v10", "va"],
}

INPUT_DIR         = Path(CONFIG["INPUT_DIR"])
OUTPUT_PARQUET    = Path(CONFIG["OUTPUT_PARQUET"])
DOMAIN            = CONFIG["DOMAIN"]
SMOOTHING_SIGMA   = CONFIG["SMOOTHING_SIGMA"]
SUBSAMPLE_STRIDE  = CONFIG["SUBSAMPLE_STRIDE"]
RAIN_MM_THRESHOLD = CONFIG["RAIN_MM_THRESHOLD"]
OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

def parse_filename(fname):
    m = re.search(CONFIG["FILENAME_REGEX"], fname)
    if not m:
        return None
    init_time = pd.Timestamp(f"{m.group(1)} {m.group(2)}:00")
    lead      = int(m.group(3))
    return {"init_time": init_time, "lead_time": lead,
            "valid_time": init_time + pd.Timedelta(hours=lead)}

def make_filename(init_time, lead):
    return CONFIG["FILENAME_TEMPLATE"].format(
        date=init_time.strftime('%Y-%m-%d'),
        hour=init_time.strftime('%H'),
        lead=lead,
    )

def standardize(ds):
    rename = {k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items() if k in ds.coords}
    return ds.rename(rename) if rename else ds

def clip_domain(ds):
    return ds.sel(lat=slice(DOMAIN["lat_min"], DOMAIN["lat_max"]),
                  lon=slice(DOMAIN["lon_min"], DOMAIN["lon_max"]))

def squeeze_time(ds):
    if "time" in ds.dims:
        ds = ds.isel(time=0, drop=True)
    return ds.drop_vars("time", errors="ignore")

def resolve_rain(ds):
    for name in VAR_MAP["rain"]:
        if name in ds:
            da = ds[name]
            if "time" in da.dims:
                da = da.isel(time=0, drop=True)
            return da.squeeze().values.astype(np.float32) * 1000
    raise KeyError(f"No rain variable found. Available: {list(ds.data_vars)}")

def get_uvlevel(ds, level):
    def extract(aliases):
        for name in aliases:
            if name not in ds:
                continue
            da = ds[name]
            if "time" in da.dims:
                da = da.isel(time=0, drop=True)
            for dim in ("level", "isobaricInhPa", "pressure_level"):
                if dim in da.dims:
                    da = da.sel({dim: level})
                    break
            return da.squeeze().values.astype(np.float32)
        return None
    u = extract(VAR_MAP["u"])
    v = extract(VAR_MAP["v"])
    if u is None or v is None:
        raise KeyError(f"u or v not found at {level} hPa")
    return u, v

def divergence_np(u, v, lat, lon):
    u_da = xr.DataArray(u.astype(float), coords={"lat": lat, "lon": lon}, dims=["lat", "lon"])
    v_da = xr.DataArray(v.astype(float), coords={"lat": lat, "lon": lon}, dims=["lat", "lon"])
    div  = mpcalc.divergence(u_da.metpy.quantify(), v_da.metpy.quantify())
    return np.array(div.metpy.dequantify(), dtype=np.float32)

def vorticity_np(u, v, lat, lon):
    R = 6371000.0
    dlat = np.deg2rad(np.abs(lat[1] - lat[0]))
    dlon = np.deg2rad(np.abs(lon[1] - lon[0]))
    dy = R * dlat
    lat_rad = np.deg2rad(lat)
    dx = R * np.cos(lat_rad) * dlon
    dvdx = np.gradient(v, axis=1) / dx[:, None]
    dudy = np.gradient(u, axis=0) / dy
    return (dvdx - dudy).astype(np.float32)

def rh_from_q_t(q, t, level_hpa):
    rh = mpcalc.relative_humidity_from_specific_humidity(
        level_hpa * 100 * munits.pascal,
        t * munits.kelvin,
        q * munits("kg/kg"),
    )
    return (rh.to("dimensionless").magnitude * 100).astype(np.float32)

def sm(arr):
    return gaussian_filter(arr, sigma=SMOOTHING_SIGMA).astype(np.float32)

def sub(arr):
    s = SUBSAMPLE_STRIDE
    return arr[::s, ::s]

def open_forecast(fpath):
    ds = xr.open_dataset(fpath, engine="netcdf4")
    ds = clip_domain(standardize(ds))
    ds = squeeze_time(ds)
    return ds

def compute_conv850(ds, lat, lon):
    u850, v850 = get_uvlevel(ds, LOW_LEVEL)
    return sm(-divergence_np(u850, v850, lat, lon))

def compute_q850(ds):
    return ds["q"].sel(level=LOW_LEVEL).squeeze().values.astype(np.float32)

def file_to_arrow(fpath, meta):
    ds  = open_forecast(fpath)
    lat = ds.lat.values
    lon = ds.lon.values

    rain = resolve_rain(ds)

    u850, v850 = get_uvlevel(ds, LOW_LEVEL)
    div850_raw = divergence_np(u850, v850, lat, lon)
    conv850    = sm(-div850_raw)
    div850     = sm(div850_raw)
    vort850    = sm(vorticity_np(u850, v850, lat, lon))
    wspd850    = sm(np.sqrt(u850**2 + v850**2))
    del u850, v850, div850_raw

    u200, v200 = get_uvlevel(ds, UPPER_LEVEL)
    div200     = sm(divergence_np(u200, v200, lat, lon))
    wspd200    = sm(np.sqrt(u200**2 + v200**2))
    del u200, v200

    try:
        u700, v700 = get_uvlevel(ds, WIND_LEVEL)
        wspd700    = sm(np.sqrt(u700**2 + v700**2))
        del u700, v700
    except Exception:
        wspd700 = np.full_like(rain, np.nan, dtype=np.float32)

    try:
        u500, v500 = get_uvlevel(ds, MID_LEVEL)
        div500     = sm(divergence_np(u500, v500, lat, lon))
        del u500, v500
    except Exception:
        div500 = np.full_like(rain, np.nan, dtype=np.float32)

    q850 = ds["q"].sel(level=LOW_LEVEL).squeeze().values.astype(np.float32)
    t850 = ds["t"].sel(level=LOW_LEVEL).squeeze().values.astype(np.float32)
    t500 = ds["t"].sel(level=MID_LEVEL).squeeze().values.astype(np.float32)
    lapse_rate = t850 - t500

    try:
        rh850 = rh_from_q_t(q850, t850, LOW_LEVEL)
    except Exception:
        rh850 = np.full_like(rain, np.nan, dtype=np.float32)

    try:
        q700 = ds["q"].sel(level=WIND_LEVEL).squeeze().values.astype(np.float32)
        t700 = ds["t"].sel(level=WIND_LEVEL).squeeze().values.astype(np.float32)
        rh700 = rh_from_q_t(q700, t700, WIND_LEVEL)
    except Exception:
        rh700 = np.full_like(rain, np.nan, dtype=np.float32)

    ds.close()

    lead      = meta["lead_time"]
    init_time = meta["init_time"]

    path_tm6 = INPUT_DIR / make_filename(init_time, lead - 6)
    if lead >= 12 and path_tm6.exists():
        try:
            ds_tm6      = open_forecast(path_tm6)
            conv850_tm6 = compute_conv850(ds_tm6, lat, lon)
            q850_tm6    = compute_q850(ds_tm6)
            ds_tm6.close()
            tend_conv850_6h = conv850 - conv850_tm6
            tend_q850_6h    = q850   - q850_tm6
        except Exception:
            tend_conv850_6h = np.zeros_like(conv850)
            tend_q850_6h    = np.zeros_like(q850)
    else:
        tend_conv850_6h = np.zeros_like(conv850)
        tend_q850_6h    = np.zeros_like(q850)

    path_tm12 = INPUT_DIR / make_filename(init_time, lead - 12)
    if lead >= 18 and path_tm12.exists():
        try:
            ds_tm12      = open_forecast(path_tm12)
            conv850_tm12 = compute_conv850(ds_tm12, lat, lon)
            ds_tm12.close()
            tend_conv850_12h = conv850 - conv850_tm12
        except Exception:
            tend_conv850_12h = np.zeros_like(conv850)
    else:
        tend_conv850_12h = np.zeros_like(conv850)

    lat_s = lat[::SUBSAMPLE_STRIDE]
    lon_s = lon[::SUBSAMPLE_STRIDE]

    arrays = dict(
        rain=rain, conv850=conv850, div850=div850, vort850=vort850,
        wspd850=wspd850, div200=div200, wspd200=wspd200, wspd700=wspd700,
        div500=div500, q850=q850, t850=t850, lapse_rate=lapse_rate,
        rh850=rh850, rh700=rh700,
        tend_conv850_6h=tend_conv850_6h, tend_conv850_12h=tend_conv850_12h,
        tend_q850_6h=tend_q850_6h,
    )
    arrays = {k: sub(v) for k, v in arrays.items()}

    lat_g, lon_g = np.meshgrid(lat_s, lon_s, indexing="ij")
    n = lat_g.size

    table = pa.table({
        "valid_time":        pa.array([meta["valid_time"]] * n),
        "init_time":         pa.array([meta["init_time"]]  * n),
        "lead_time_hr":      pa.array(np.full(n, lead, np.int16)),
        "lat":               pa.array(lat_g.ravel().astype(np.float32)),
        "lon":               pa.array(lon_g.ravel().astype(np.float32)),
        "P_mm":              pa.array(arrays["rain"].ravel()),
        "ln_P_eps":          pa.array(np.log(arrays["rain"].ravel() + EPS)),
        "is_raining":        pa.array((arrays["rain"].ravel() >= RAIN_MM_THRESHOLD).astype(np.int8)),
        "CONV_850":          pa.array(arrays["conv850"].ravel()),
        "DIV_850":           pa.array(arrays["div850"].ravel()),
        "VORT_850":          pa.array(arrays["vort850"].ravel()),
        "WSPD_850":          pa.array(arrays["wspd850"].ravel()),
        "DIV_200":           pa.array(arrays["div200"].ravel()),
        "WSPD_200":          pa.array(arrays["wspd200"].ravel()),
        "WSPD_700":          pa.array(arrays["wspd700"].ravel()),
        "DIV_500":           pa.array(arrays["div500"].ravel()),
        "q_850":             pa.array(arrays["q850"].ravel()),
        "T_850":             pa.array(arrays["t850"].ravel()),
        "lapse_rate":        pa.array(arrays["lapse_rate"].ravel()),
        "RH_850":            pa.array(arrays["rh850"].ravel()),
        "RH_700":            pa.array(arrays["rh700"].ravel()),
        "CONV850_x_q850":    pa.array(arrays["conv850"].ravel() * arrays["q850"].ravel()),
        "TEND_CONV850_6h":   pa.array(arrays["tend_conv850_6h"].ravel()),
        "TEND_CONV850_12h":  pa.array(arrays["tend_conv850_12h"].ravel()),
        "TEND_q850_6h":      pa.array(arrays["tend_q850_6h"].ravel()),
    })

    return table

if __name__ == "__main__":
    nc_files = sorted(INPUT_DIR.glob("*.nc"))
    print(f"Found {len(nc_files)} .nc files")
    print(f"Output: {OUTPUT_PARQUET}")
    print(f"Subsample stride: {SUBSAMPLE_STRIDE}\n")

    done_times = set()
    if OUTPUT_PARQUET.exists():
        existing   = pq.read_table(OUTPUT_PARQUET, columns=["valid_time"])
        done_times = set(existing["valid_time"].to_pylist())
        print(f"Resuming -- {len(done_times)} valid_times already written\n")
        del existing

    writer = None
    stats  = {"ok": 0, "skipped": 0, "total_rows": 0}

    try:
        for i, fpath in enumerate(nc_files):
            meta = parse_filename(fpath.name)
            if meta is None:
                print(f"  [SKIP] pattern mismatch: {fpath.name}")
                stats["skipped"] += 1
                continue
            if meta["valid_time"] in done_times:
                stats["skipped"] += 1
                continue

            print(f"  [{i+1:>4}/{len(nc_files)}] {fpath.name} ...", end=" ", flush=True)
            try:
                table = file_to_arrow(fpath, meta)
            except Exception as e:
                print(f"FAIL -- {e}")
                stats["skipped"] += 1
                continue

            if writer is None:
                writer = pq.ParquetWriter(OUTPUT_PARQUET, table.schema, compression="snappy")

            writer.write_table(table)
            n = len(table)
            del table
            gc.collect()

            stats["ok"]         += 1
            stats["total_rows"] += n
            print(f"OK  ({n:,} rows, cumulative: {stats['total_rows']:,})")
    finally:
        if writer is not None:
            writer.close()
            size_mb = OUTPUT_PARQUET.stat().st_size / 1e6
            print(f"\nWriter closed. File size: {size_mb:.1f} MB")

    print(f"\n{'='*55}")
    print(f"Files OK      : {stats['ok']}")
    print(f"Files skipped : {stats['skipped']}")
    print(f"Total rows    : {stats['total_rows']:,}")
    print(f"{'='*55}")
