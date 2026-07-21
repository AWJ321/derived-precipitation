#!/usr/bin/env python3
import warnings
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pickle
from pathlib import Path

warnings.filterwarnings("ignore")

CONFIG = {
    "PARQUET_FILE": "/path/to/precip_predictors.parquet",
    "OUTPUT_DIR": "/path/to/output/models",
    "MODEL_TYPE": "mlp",
    "VAL_FRACTION": 0.15,
    "CHUNK_SIZE": 500_000,
    "LGBM_PARAMS": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": 4,
    },
    "MLP_PARAMS": {
        "batch_size": 4096,
        "max_epochs": 100,
        "learning_rate": 0.001,
        "patience": 10,
    },
}

EPS = 0.1
PREDICTORS_9 = [
    "CONV_850", "DIV_200", "q_850", "T_850", "WSPD_700",
    "CONV850_x_q850", "TEND_CONV850_6h", "TEND_CONV850_12h", "TEND_q850_6h",
]
PREDICTORS_15 = PREDICTORS_9 + [
    "RH_700", "lapse_rate", "WSPD_850", "VORT_850", "WSPD_200", "DIV_500",
]
SEASONS = {
    "NE": [11, 12, 1, 2, 3],
    "SW": [6, 7, 8, 9],
    "IM": [4, 5, 10],
}

PARQUET_FILE = Path(CONFIG["PARQUET_FILE"])
OUT_DIR      = Path(CONFIG["OUTPUT_DIR"])
MODEL_TYPE   = CONFIG["MODEL_TYPE"].lower()
PREDICTORS   = PREDICTORS_9 if MODEL_TYPE == "ols" else PREDICTORS_15
VAL_FRACTION = CONFIG["VAL_FRACTION"]
CHUNK_SIZE   = CONFIG["CHUNK_SIZE"]
OUT_DIR.mkdir(parents=True, exist_ok=True)

if MODEL_TYPE == "lgbm":
    import lightgbm as lgb
elif MODEL_TYPE == "mlp":
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

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

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def load_season_data(months):
    pf = pq.ParquetFile(PARQUET_FILE)
    dfs = []
    for batch in pf.iter_batches(batch_size=CHUNK_SIZE,
                                  columns=["is_raining", "ln_P_eps", "valid_time"] + PREDICTORS):
        df = batch.to_pandas()
        df["month"] = pd.to_datetime(df["valid_time"]).dt.month
        df = df[df["month"].isin(months) & (df["is_raining"] == 1)]
        df = df.dropna(subset=PREDICTORS + ["ln_P_eps"])
        if not df.empty:
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def fit_ols(season_name, df_tr, df_va):
    X_tr = np.column_stack([np.ones(len(df_tr))] + [df_tr[p].values.astype(np.float64) for p in PREDICTORS])
    y_tr = df_tr["ln_P_eps"].values.astype(np.float64)
    X_va = np.column_stack([np.ones(len(df_va))] + [df_va[p].values.astype(np.float64) for p in PREDICTORS])
    y_va = df_va["ln_P_eps"].values.astype(np.float64)

    XtX_tr, Xty_tr = X_tr.T @ X_tr, X_tr.T @ y_tr
    coeffs_tr = np.linalg.solve(XtX_tr, Xty_tr)

    y_pred = X_va @ coeffs_tr
    print(f"  Val R2 (log): {r2(y_va, y_pred):.4f}")

    coeff_df = pd.DataFrame({
        "predictor": ["intercept"] + PREDICTORS,
        "coeff_train": coeffs_tr,
    })
    out_path = OUT_DIR / f"coefficients_{season_name}.csv"
    coeff_df.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")

def fit_lgbm(season_name, df_tr, df_va):
    X_tr = df_tr[PREDICTORS].values.astype(np.float32)
    y_tr = df_tr["ln_P_eps"].values.astype(np.float32)
    X_va = df_va[PREDICTORS].values.astype(np.float32)
    y_va = df_va["ln_P_eps"].values.astype(np.float32)

    model = lgb.LGBMRegressor(**CONFIG["LGBM_PARAMS"], verbose=-1)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )

    y_pred = model.predict(X_va)
    print(f"  Best iteration: {model.best_iteration_}")
    print(f"  Val R2 (log)  : {r2(y_va, y_pred):.4f}")

    out_path = OUT_DIR / f"lgbm_model_{season_name}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved -> {out_path}")

def fit_mlp(season_name, df_tr, df_va):
    p = CONFIG["MLP_PARAMS"]
    X_tr = df_tr[PREDICTORS].values.astype(np.float32)
    y_tr = df_tr["ln_P_eps"].values.astype(np.float32)
    X_va = df_va[PREDICTORS].values.astype(np.float32)
    y_va = df_va["ln_P_eps"].values.astype(np.float32)

    mean = X_tr.mean(axis=0)
    std  = X_tr.std(axis=0)
    std[std == 0] = 1.0
    X_tr = (X_tr - mean) / std
    X_va = (X_va - mean) / std

    scaler = {"mean": mean, "std": std}
    with open(OUT_DIR / f"mlp_scaler_{season_name}.pkl", "wb") as f:
        pickle.dump(scaler, f)

    tr_dl = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                       batch_size=p["batch_size"], shuffle=True, num_workers=0)
    va_dl = DataLoader(TensorDataset(torch.tensor(X_va), torch.tensor(y_va)),
                       batch_size=p["batch_size"], shuffle=False, num_workers=0)

    model     = PrecipMLP(len(PREDICTORS)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
    criterion = nn.MSELoss()

    best_val_loss    = float("inf")
    patience_counter = 0
    best_state       = None
    model_out        = OUT_DIR / f"mlp_model_{season_name}.pt"

    for epoch in range(p["max_epochs"]):
        model.train()
        for X_batch, y_batch in tr_dl:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(X_batch), y_batch).backward()
            optimizer.step()

        model.eval()
        va_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in va_dl:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                va_loss += criterion(model(X_batch), y_batch).item() * len(X_batch)
        va_loss /= len(df_va)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:>3}/{p['max_epochs']}  va_loss={va_loss:.4f}", flush=True)

        if va_loss < best_val_loss:
            best_val_loss    = va_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, model_out)
        else:
            patience_counter += 1
            if patience_counter >= p["patience"]:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in va_dl:
            all_preds.append(model(X_batch.to(DEVICE)).cpu().numpy())
    y_pred = np.concatenate(all_preds)
    print(f"  Val R2: {r2(y_va, y_pred):.4f}")
    print(f"  Saved -> {model_out}")

def fit_season(season_name, months):
    print(f"\n{'='*55}")
    print(f"Season: {season_name}  (months {months})")
    print(f"Model : {MODEL_TYPE.upper()}")
    print(f"{'='*55}")

    df_all = load_season_data(months)
    n_train = int(len(df_all) * (1 - VAL_FRACTION))
    df_tr = df_all.iloc[:n_train]
    df_va = df_all.iloc[n_train:]
    print(f"  Wet train rows: {len(df_tr):,}  |  Wet val rows: {len(df_va):,}")

    if MODEL_TYPE == "ols":
        fit_ols(season_name, df_tr, df_va)
    elif MODEL_TYPE == "lgbm":
        fit_lgbm(season_name, df_tr, df_va)
    elif MODEL_TYPE == "mlp":
        fit_mlp(season_name, df_tr, df_va)
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE} (expected ols/lgbm/mlp)")

if __name__ == "__main__":
    print(f"Parquet    : {PARQUET_FILE}")
    print(f"Output     : {OUT_DIR}")
    print(f"Model type : {MODEL_TYPE.upper()}")
    for season_name, months in SEASONS.items():
        fit_season(season_name, months)
    print("\nDone.")
