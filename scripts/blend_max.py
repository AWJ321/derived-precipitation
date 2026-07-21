#!/usr/bin/env python3
import warnings
import numpy as np
import xarray as xr
from pathlib import Path

warnings.filterwarnings("ignore")

CONFIG = {
    "INPUT_DIR_A": "/path/to/derived/ols_output",
    "INPUT_DIR_B": "/path/to/derived/lgbm_output",
    "OUTPUT_DIR": "/path/to/derived/blend_output",
    "FILENAME_GLOB": "derived_*.nc",
    "VAR_NAME": "precip_derived",
}

INPUT_DIR_A = Path(CONFIG["INPUT_DIR_A"])
INPUT_DIR_B = Path(CONFIG["INPUT_DIR_B"])
OUTPUT_DIR  = Path(CONFIG["OUTPUT_DIR"])
VAR_NAME    = CONFIG["VAR_NAME"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    files_a = sorted(INPUT_DIR_A.glob(CONFIG["FILENAME_GLOB"]))
    print(f"Files in INPUT_DIR_A: {len(files_a)}")

    ok = skipped = failed = 0
    for path_a in files_a:
        out_path = OUTPUT_DIR / path_a.name
        if out_path.exists():
            skipped += 1
            continue

        path_b = INPUT_DIR_B / path_a.name
        if not path_b.exists():
            print(f"  Matching file not found in INPUT_DIR_B: {path_a.name}")
            failed += 1
            continue

        try:
            arr_a = xr.open_dataset(path_a)[VAR_NAME].values.astype(np.float32)
            ds_b  = xr.open_dataset(path_b)
            arr_b = ds_b[VAR_NAME].values.astype(np.float32)

            blended = np.maximum(arr_a, arr_b)

            ds_out = xr.Dataset(
                {VAR_NAME: (["lat", "lon"], blended)},
                coords={"lat": ds_b.lat.values, "lon": ds_b.lon.values}
            )
            ds_out[VAR_NAME].attrs = {
                "units": "mm",
                "long_name": "Derived 6-hourly precipitation (max blend)",
                "method": "max(input_A, input_B)"
            }
            ds_out.attrs = ds_b.attrs
            ds_b.close()
            ds_out.to_netcdf(out_path)
            ok += 1
        except Exception as e:
            print(f"  FAIL {path_a.name}: {e}")
            failed += 1

    print(f"\nDone. OK: {ok}  Skipped: {skipped}  Failed: {failed}")
