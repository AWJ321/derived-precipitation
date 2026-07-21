# Derived Precipitation for AI-NWP Models

Statistically derives 6-hourly precipitation from any AI-NWP model's
atmospheric forecast fields, using one of three pretrained models
(OLS, LightGBM, or MLP) fitted on AIFS forecasts and Southeast Asia
GPM IMERG observations.

## Why this exists

Many AI-NWP models (e.g. FourCastNet, Pangu-Weather, Aurora) do not
produce precipitation natively. This pipeline derives a precipitation
estimate from the atmospheric fields such models *do* produce
(winds, humidity, temperature, geopotential), using a model that was
fitted once on AIFS data and then applied to any other model's fields
at inference time.

## Pipeline overview
- **Most users only need `derive_precipitation.py`** (+ `blend_max.py`
  if using LightGBM) — the pretrained models are already provided in
  `models/`, ready to apply to your own NWP model's fields.
- **`extract_predictors.py` and `fit_model.py`** are provided for
  transparency and reproducibility — they document and let you rerun
  exactly how the models in `models/` were trained. They are
  intentionally AIFS-specific (see "Important limitation" below).

## Requirements for your own NWP model's output (for `derive_precipitation.py`)

Your forecast files must provide the following variables, at the
listed pressure levels, in a format `xarray.open_dataset()` can read
(GRIB, NetCDF, HDF5, Zarr, etc. — set `ENGINE` in the config
accordingly):

| Variable | Level(s) | Notes |
|---|---|---|
| u, v (wind components) | 850, 700, 500, 200 hPa | 700/500 hPa are optional — see fallback behaviour below |
| Specific humidity (q) OR relative humidity (r) | 850, 700 hPa | either is accepted; RH is converted internally |
| Temperature (t) | 850, 500 hPa | |
| Coordinates | lat, lon (or latitude/longitude) | any range/ordering is handled automatically |

**Lead time coverage:** the pipeline needs, at minimum, files for
every 6-hour lead you want to derive (e.g. 6, 12, 18h...). If your
model has no lead=0 output, tendency terms will be zero at your two
earliest available leads and real thereafter — this does not require
any code changes, it happens automatically.

**700/500 hPa fallback:** if your model doesn't output these levels,
the corresponding predictors (`WSPD_700`, `DIV_500`) default to zero
rather than failing. This will somewhat degrade prediction quality
compared to having the full variable set. Note: OLS only uses 9 of
the 15 predictors (it does not use `RH_700`, `lapse_rate`,
`WSPD_850`, `VORT_850`, `WSPD_200`, `DIV_500`), so this fallback is
only relevant for LightGBM/MLP.

## Model file formats expected in `MODEL_DIR`

- **OLS**: `coefficients_{season}.csv`, with columns `predictor` and
  `coeff_train`, including a row where `predictor == "intercept"`
- **LightGBM**: `lgbm_model_{season}.pkl` (pickled booster)
- **MLP**: `mlp_model_{season}.pt` (PyTorch state dict) +
  `mlp_scaler_{season}.pkl` (dict with `mean`/`std` keys)

where `{season}` is one of `NE`, `SW`, `IM`.

## Install

```bash
pip install -r requirements.txt
```

(`requirements.txt` marks which packages are only needed for specific
`ENGINE`/`MODEL_TYPE` choices — you don't need all of them.)

---

## 1. `derive_precipitation.py` — apply a fitted model to any NWP model's fields

Edit the `CONFIG` block, then run:
```bash
python3 scripts/derive_precipitation.py
```
Output is one NetCDF file per (init_time, lead), each containing a
single `precip_derived` variable in mm.

**Example config** (applying the pretrained MLP model to FourCastNet's
GRIB output):
```python
CONFIG = {
    "INPUT_DIR": "/data/fourcastnet/forecasts",
    "FILENAME_GLOB": "fv2_merged_2024-*.grib",
    "FILENAME_REGEX": r"(\d{4}-\d{2}-\d{2})_(\d{2})-out-(\d+)",
    "FILENAME_TEMPLATE": "fv2_merged_{date}_{hour}-out-{lead}.grib",
    "ENGINE": "cfgrib",
    "DOMAIN": {"lat_min": -12, "lat_max": 23, "lon_min": 92, "lon_max": 127},
    "MODEL_TYPE": "mlp",
    "MODEL_DIR": "models",
    "SMOOTHING_SIGMA": 1,
    "OUTPUT_DIR": "/data/fourcastnet/derived_mlp",
    "YEAR_FILTER": 2024,
}
```

## 2. `blend_max.py` — only needed for LightGBM

LightGBM needs its output combined with OLS's via elementwise maximum
to handle heavy rain reliably (MLP does not need this step — see
"Which model should I use?" below). Run `derive_precipitation.py`
twice — once with `MODEL_TYPE="ols"`, once with `MODEL_TYPE="lgbm"` —
then blend the two output folders:

```bash
python3 scripts/blend_max.py
```

**Example config:**
```python
CONFIG = {
    "INPUT_DIR_A": "/data/fourcastnet/derived_ols",
    "INPUT_DIR_B": "/data/fourcastnet/derived_lgbm",
    "OUTPUT_DIR": "/data/fourcastnet/derived_blend",
    "FILENAME_GLOB": "derived_*.nc",
    "VAR_NAME": "precip_derived",
}
```

## 3. `extract_predictors.py` — build training data from AIFS + GPM

Builds the parquet file `fit_model.py` reads. Intentionally
AIFS-specific (see "Important limitation" below).

```bash
python3 scripts/extract_predictors.py
```

**Example config:**
```python
CONFIG = {
    "INPUT_DIR": "/data/aifs/forecasts_regridded",
    "FILENAME_REGEX": r"(\d{4}-\d{2}-\d{2})_(\d{2})-out-(\d+)",
    "FILENAME_TEMPLATE": "aifs_gridded_{date}_{hour}-out-{lead}_regridded.nc",
    "DOMAIN": {"lat_min": -12, "lat_max": 23, "lon_min": 92, "lon_max": 127},
    "SMOOTHING_SIGMA": 10,
    "SUBSAMPLE_STRIDE": 4,
    "RAIN_MM_THRESHOLD": 0.1,
    "OUTPUT_PARQUET": "output/precip_predictors_aifs.parquet",
}
```

## 4. `fit_model.py` — train OLS, LightGBM, or MLP

Reads the parquet from step 3, fits one model per season (NE/SW/IM).

```bash
python3 scripts/fit_model.py
```

**Example config** (fitting MLP):
```python
CONFIG = {
    "PARQUET_FILE": "output/precip_predictors_aifs.parquet",
    "OUTPUT_DIR": "output/models_mlp",
    "MODEL_TYPE": "mlp",
    "VAL_FRACTION": 0.15,
    "CHUNK_SIZE": 500_000,
    "LGBM_PARAMS": {
        "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 63,
        "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8,
        "n_jobs": 4,
    },
    "MLP_PARAMS": {
        "batch_size": 4096, "max_epochs": 100,
        "learning_rate": 0.001, "patience": 10,
    },
}
```
(`LGBM_PARAMS`/`MLP_PARAMS` are only used by the matching `MODEL_TYPE`
— set `MODEL_TYPE: "lgbm"` or `"ols"` to fit the other two instead.)

---

## Which model should I use?

- **MLP** — recommended default. Best overall performance across
  every AI-NWP target we tested it on (FourCastNet, Pangu-Weather,
  Aurora, GraphCast), and does not need any post-processing blend.
- **LightGBM** — statistically comparable to MLP in most cases, but
  needs a max-blend with the OLS output (see `blend_max.py`) to
  handle heavy rain reliably on its own.
- **OLS** — simplest, fastest, no dependencies beyond numpy — useful
  as a baseline or as the blend partner for LightGBM.

## Important limitation — regional specificity

All three pretrained models were fitted on **Southeast Asia**
precipitation and atmospheric regimes (seasonally stratified into
Northeast monsoon, Southwest monsoon, and Intermonsoon periods).
Applying them to a different climate region will run without error,
but the physical validity of the output for that region has **not**
been evaluated and should not be assumed. Retraining on
region-appropriate data is recommended for use outside Southeast Asia.

`extract_predictors.py` is intentionally AIFS-specific rather than
configurable to any source model — the methodology's validity relies
on predictor and target coming from the same model for internal
consistency (see manuscript). Retraining on a different source model
would require adapting the variable-reading logic itself, not just
the config.

## Training data

The full training dataset (extracted AIFS predictors + GPM
precipitation targets, ~1.5GB) is included in this repo via Git LFS
(`precip_predictors_aifs.parquet`). Run `git lfs pull` after
cloning to fetch it, or see `extract_predictors.py` to regenerate
it yourself from raw AIFS forecasts.st your own AIFS forecast archive to
regenerate it.
