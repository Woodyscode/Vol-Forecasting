#!/usr/bin/env python3
import warnings; warnings.filterwarnings("ignore")
import os
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# PyTorch (for LSTM)
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%H:%M:%S"
)

# -------- Config --------
@dataclass
class VolConfig:
    tickers: List[str] = None
    start: str = "2018-01-01"
    end: str = None
    price_col: str = "Adj Close"
    data_dir: str = "data/raw"
    h_forecast: int = 5
    annualization: int = 252
    dist: str = "t"
    mean: str = "constant"
    vol: str = "GARCH"
    p: int = 1; o: int = 0; q: int = 1
    realized_window: int = 21

    # LSTM (PyTorch)
    lstm_lookback: int = 30
    lstm_units: int = 64
    lstm_dropout: float = 0.2
    lstm_epochs: int = 30
    lstm_batch_size: int = 32
    lstm_train_frac: float = 0.8
    lstm_lr: float = 1e-3

    # Walk-forward backtest controls
    wf_min_obs: int = 400        # minimum history before first OOS prediction
    wf_retrain_every: int = 5    # retrain frequency (1 = daily; 5 ~ weekly)


# -------- Data utils --------
def fetch_prices(tickers, start, end, price_col="Adj Close", save_dir=None):
    import os
    os.makedirs(save_dir, exist_ok=True)

    all_prices = pd.DataFrame()

    for ticker in tickers:
        file_path = os.path.join(save_dir, f"{ticker}.csv")

        # 1. Load from cache or download
        if os.path.exists(file_path):
            logging.info(f"[{ticker}] Loading cached data from {file_path}")
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        else:
            logging.info(f"[{ticker}] Downloading new data from Yahoo Finance...")
            df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
            df.to_csv(file_path)
            logging.info(f"[{ticker}] Saved raw data to {file_path}")

        # 2. Handle column structure (MultiIndex or single)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(price_col, axis=1, level=0)
        if isinstance(df, pd.DataFrame) and df.shape[1] == 1:
            df.columns = [ticker]
        elif isinstance(df, pd.Series):
            df = df.to_frame(name=ticker)
        elif price_col in df.columns:
            df = df[[price_col]].rename(columns={price_col: ticker})
        else:
            raise ValueError(f"{price_col} not found in {ticker} data")

        # 3. Clean numeric data
        df[ticker] = pd.to_numeric(df[ticker], errors="coerce")
        df = df.dropna(subset=[ticker])
        df = df.sort_index()

        # 4. Append to main frame
        all_prices = pd.concat([all_prices, df], axis=1)

    return all_prices.dropna(how="all")

def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="all")

def annualize_sigma_from_percent(sigma_percent: pd.Series, annualization: int) -> pd.Series:
    return (sigma_percent / 100.0) * math.sqrt(annualization)

def annualize_variance_from_percent2(var_percent2: pd.Series, annualization: int) -> pd.Series:
    return (np.sqrt(var_percent2) / 100.0) * math.sqrt(annualization)


# -------- GARCH fit & forecast --------
def fit_garch_and_forecast(returns: pd.Series, cfg: VolConfig
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, object]:
    r_pct = 100 * returns.dropna()
    res = arch_model(r_pct, mean=cfg.mean, vol=cfg.vol, p=cfg.p, o=cfg.o, q=cfg.q,
                     dist=cfg.dist, rescale=False).fit(disp="off")
    sigma_in_pct = res.conditional_volatility.rename("sigma_in_pct")
    f = res.forecast(horizon=cfg.h_forecast, reindex=True)
    var_1d = f.variance.dropna(how="all").iloc[:, 0].rename("var_forecast_1d_pct2")
    vol_in_annual  = annualize_sigma_from_percent(sigma_in_pct, cfg.annualization).rename("garch_in_sample_vol_annual")
    vol_fcast_ann  = annualize_variance_from_percent2(var_1d, cfg.annualization).rename("garch_forecast_vol_annual")
    return sigma_in_pct, var_1d, vol_in_annual, vol_fcast_ann, res

def realized_vol(returns: pd.Series, window=21, annualization=252) -> pd.Series:
    return (returns.rolling(window).std() * math.sqrt(annualization)).rename("realized_vol_annual")


# -------- Plotting (now includes LSTM lines & OOS GARCH) --------
def plot_vol_series_full(
    ticker: str,
    realized_annual: pd.Series,
    in_sample_annual: pd.Series,
    forecast_annual: pd.Series | None = None,
    oos_lines: Dict[str, pd.Series] | None = None,
) -> None:
    plt.figure(figsize=(11, 6))
    realized_annual.dropna().plot(label="Realized Vol (annualized)")
    in_sample_annual.dropna().plot(label="GARCH In-sample (annualized)")
    if forecast_annual is not None and not forecast_annual.dropna().empty:
        forecast_annual.dropna().plot(label="GARCH 1D Forecast (in-sample)", linestyle="--")
    if oos_lines:
        for lbl, s in oos_lines.items():
            if s is not None and not s.dropna().empty:
                s.dropna().plot(label=lbl, linestyle=":")
    plt.title(f"{ticker} — Realized vs. GARCH & LSTM")
    plt.ylabel("Annualized Volatility (decimal)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# -------- Accuracy (same-day, aligned) --------
def accuracy_same_day(model_vol: pd.Series, realized_vol_annual: pd.Series) -> dict:
    df = pd.concat([model_vol.rename("model"), realized_vol_annual.rename("real")], axis=1).dropna()
    if df.empty:
        return {"RMSE": np.nan, "MAE": np.nan, "MAPE(%)": np.nan, "R2": np.nan, "n_obs": 0}
    rmse = float(np.sqrt(mean_squared_error(df["real"], df["model"])))
    mae  = float(mean_absolute_error(df["real"], df["model"]))
    mape = float((np.abs((df["real"] - df["model"]) / df["real"]).mean()) * 100)
    r2   = float(r2_score(df["real"], df["model"]))
    return {"RMSE": rmse, "MAE": mae, "MAPE(%)": mape, "R2": r2, "n_obs": int(len(df))}


# -------- LSTM (PyTorch) — simple train/test split (kept) --------
class RVDataset(Dataset):
    def __init__(self, series: pd.Series, lookback: int, scaler: MinMaxScaler):
        vals = series.values.reshape(-1, 1)
        self.scaler = scaler
        self.scaled = scaler.fit_transform(vals)   # NOTE: simple approach uses full fit
        self.lookback = lookback
    def __len__(self): return max(0, len(self.scaled) - self.lookback)
    def __getitem__(self, idx):
        x = self.scaled[idx: idx+self.lookback]
        y = self.scaled[idx+self.lookback]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class LSTMRV(nn.Module):
    def __init__(self, input_size=1, hidden=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.drop(out)
        return self.fc(out)

def fit_predict_lstm_rv(rv_series: pd.Series, cfg: VolConfig):
    rv = rv_series.dropna().copy()
    if len(rv) < cfg.lstm_lookback + 40:
        raise ValueError("Not enough data for LSTM windowing.")

    scaler = MinMaxScaler()
    dataset = RVDataset(rv, cfg.lstm_lookback, scaler)
    n = len(dataset)
    split = int(cfg.lstm_train_frac * n)
    train_ds = torch.utils.data.Subset(dataset, range(0, split))
    test_ds  = torch.utils.data.Subset(dataset, range(split, n))

    train_loader = DataLoader(train_ds, batch_size=cfg.lstm_batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.lstm_batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRV(hidden=cfg.lstm_units, dropout=cfg.lstm_dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lstm_lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(cfg.lstm_epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()

    # Predict on test
    model.eval()
    preds_scaled, y_true_scaled = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds_scaled.append(model(xb.to(device)).cpu().numpy())
            y_true_scaled.append(yb.numpy())
    preds_scaled = np.vstack(preds_scaled).reshape(-1, 1)
    y_true_scaled = np.vstack(y_true_scaled).reshape(-1, 1)

    # Inverse scale to annualized decimals
    y_pred = scaler.inverse_transform(preds_scaled).ravel()
    y_true = scaler.inverse_transform(y_true_scaled).ravel()

    # Dates aligned to test targets
    test_start_idx = cfg.lstm_lookback + split
    dates = rv.index[test_start_idx: test_start_idx + len(y_true)]
    return (
        pd.Series(y_pred, index=dates, name="lstm_pred_vol_annual"),
        pd.Series(y_true, index=dates, name="lstm_true_vol_annual"),
        model,
    )


# -------- LSTM Walk-forward Backtest (train 80 / val 20 inside window) --------
def _make_seq(arr_2d: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(lookback, len(arr_2d)):
        X.append(arr_2d[i - lookback:i, :])
        y.append(arr_2d[i, :])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

def _train_lstm_once(rv_window: pd.Series, cfg: VolConfig, device: torch.device):
    vals = rv_window.values.reshape(-1, 1).astype(np.float32)
    if len(vals) < cfg.lstm_lookback + 20:
        raise ValueError("Not enough data in window for LSTM.")

    X_all, y_all = _make_seq(vals, cfg.lstm_lookback)
    n = len(X_all)
    split = max(int(0.8 * n), 1)

    X_tr, y_tr = X_all[:split], y_all[:split]
    X_va, y_va = X_all[split:], y_all[split:]

    scaler = MinMaxScaler()
    scaler.fit(y_tr)  # fit on train targets only (1 feature)
    X_tr_s = scaler.transform(X_tr.reshape(-1, 1)).reshape(X_tr.shape)
    y_tr_s = scaler.transform(y_tr)
    X_va_s = scaler.transform(X_va.reshape(-1, 1)).reshape(X_va.shape) if len(X_va) else X_va
    y_va_s = scaler.transform(y_va) if len(y_va) else y_va

    tr_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_tr_s), torch.from_numpy(y_tr_s))
    tr_loader = DataLoader(tr_ds, batch_size=cfg.lstm_batch_size, shuffle=False)

    model = LSTMRV(hidden=cfg.lstm_units, dropout=cfg.lstm_dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lstm_lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(cfg.lstm_epochs):
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()

    return model, scaler

def backtest_lstm_full(
    rv_series: pd.Series,
    cfg: VolConfig,
    min_obs: int,
    retrain_every: int,
    verbose: bool = False,
) -> tuple[pd.Series, pd.Series]:
    rv = rv_series.dropna().copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preds = {}
    T = len(rv)
    start_idx = max(min_obs, cfg.lstm_lookback + 1)

    k = start_idx
    while k < T:
        rv_window = rv.iloc[:k]  # strictly up to k-1
        try:
            model, scaler = _train_lstm_once(rv_window, cfg, device)
        except Exception as e:
            if verbose: print(f"[LSTM WF] window end idx {k} failed: {e}")
            k += retrain_every
            continue

        model.eval()
        # roll out next 'retrain_every' days with same model
        for j in range(retrain_every):
            t_idx = k + j
            if t_idx >= T: break
            if t_idx - cfg.lstm_lookback < 0: continue
            seq = rv.iloc[t_idx - cfg.lstm_lookback: t_idx].values.reshape(-1, 1).astype(np.float32)
            seq_s = scaler.transform(seq).reshape(1, cfg.lstm_lookback, 1)
            with torch.no_grad():
                yhat_s = model(torch.from_numpy(seq_s).to(device)).cpu().numpy()
            yhat = scaler.inverse_transform(yhat_s).ravel()[0]
            preds[rv.index[t_idx]] = float(yhat)

        if verbose and (k - start_idx) % 50 == 0:
            print(f"[LSTM WF] progressed to {rv.index[min(k, T-1)].date()} — {len(preds)} preds")
        k += retrain_every

    pred_series = pd.Series(preds, name="lstm_bt_pred_vol_annual").sort_index()
    true_series = rv.reindex(pred_series.index).rename("lstm_bt_true_vol_annual")
    return pred_series, true_series


# -------- True OOS forecaster (GARCH only; we’re comparing vs LSTM) --------
def oos_dynamic_vol(
    returns: pd.Series,
    target_dates: pd.DatetimeIndex,
    cfg: VolConfig,
    vol_type: str = "GARCH",
    p: int = 1, o: int = 0, q: int = 1,
    dist: str | None = None,
    min_obs: int = 400,
    verbose: bool = False,
) -> pd.Series:
    r_pct_full = (100 * returns.dropna()).copy()
    out = {}
    for i, d in enumerate(pd.to_datetime(target_dates)):
        ins = r_pct_full[r_pct_full.index < d]
        if len(ins) < min_obs:
            out[d] = np.nan
            continue
        am = arch_model(ins, mean=cfg.mean, vol=vol_type, p=p, o=o, q=q, dist=(dist or cfg.dist), rescale=False)
        res = am.fit(disp="off")
        f = res.forecast(horizon=1, reindex=False)
        var_next_pct2 = float(f.variance.iloc[-1, 0])
        out[d] = (np.sqrt(var_next_pct2) / 100.0) * np.sqrt(cfg.annualization)
        if verbose and (i + 1) % 50 == 0:
            print(f"[GARCH OOS] {i+1}/{len(target_dates)} processed")
    return pd.Series(out, name="garch_oos_vol_annual").sort_index()

def dm_test(y_true: pd.Series,
            y_pred_A: pd.Series,
            y_pred_B: pd.Series,
            h: int = 1,
            power: int = 2,
            lag: int | None = None,
            alternative: str = "two-sided",
            harvey_adj: bool = True) -> dict:
    """
    Diebold–Mariano test comparing forecast A vs B on the SAME dates.
    Loss is |e|^power (power=2 => MSE). p-value uses asymptotic N(0,1).
    """
    df = pd.concat(
        [y_true.rename("y"), y_pred_A.rename("A"), y_pred_B.rename("B")],
        axis=1
    ).dropna()
    if df.empty:
        return {"stat": np.nan, "p_value": np.nan, "T": 0}

    eA = df["y"] - df["A"]
    eB = df["y"] - df["B"]
    if power == 1:
        d = np.abs(eA) - np.abs(eB)
    elif power == 2:
        d = eA**2 - eB**2
    else:
        d = np.abs(eA)**power - np.abs(eB)**power

    T = len(d)
    d_bar = d.mean()

    # Newey–West variance of d with Bartlett weights
    if lag is None:
        lag = max(h - 1, 0)
    gamma0 = np.var(d, ddof=1)
    var = gamma0
    for L in range(1, lag + 1):
        w = 1.0 - L / (lag + 1)
        cov = np.cov(d[L:], d[:-L], ddof=1)[0, 1]
        var += 2 * w * cov
    if var <= 0 or np.isnan(var):
        return {"stat": np.nan, "p_value": np.nan, "T": T}

    dm = d_bar / np.sqrt(var / T)

    # Harvey–Leybourne–Newbold small-sample adjustment (optional)
    if harvey_adj and h > 1:
        k = np.sqrt((T + 1 - 2 * h + (h * (h - 1)) / T) / T)
        dm *= k

    # Normal p-value
    Phi = lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    if alternative == "two-sided":
        p = 2.0 * min(Phi(dm), 1.0 - Phi(dm))
    elif alternative == "greater":     # A beats B (smaller loss)
        p = 1.0 - Phi(dm)
    else:                               # "less": A worse than B
        p = Phi(dm)

    return {"stat": float(dm), "p_value": float(p), "T": int(T)}

def har_rv_oos(rv: pd.Series,
               target_dates: pd.DatetimeIndex,
               min_obs: int = 60) -> pd.Series:
    """
    HAR-RV: RV_t = b0 + b1*RV_{t-1} + b2*mean(RV_{t-5:t-1}) + b3*mean(RV_{t-22:t-1})
    Daily refit on data strictly before each target date. Returns annualized decimals.
    """
    rv = rv.dropna().copy()
    df = pd.DataFrame(index=rv.index)
    df["y"]   = rv
    df["rv1"] = rv.shift(1)
    df["rv5"] = rv.rolling(5).mean().shift(1)
    df["rv22"]= rv.rolling(22).mean().shift(1)

    preds = {}
    for d in pd.to_datetime(target_dates):
        if d not in df.index:
            preds[d] = np.nan
            continue
        ins = df.index < d
        sub = df.loc[ins, ["y", "rv1", "rv5", "rv22"]].dropna()
        if len(sub) < min_obs:
            preds[d] = np.nan
            continue

        X = np.c_[np.ones(len(sub)), sub[["rv1", "rv5", "rv22"]].values]
        y = sub["y"].values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)

        x_d = df.loc[d, ["rv1", "rv5", "rv22"]]
        if x_d.isna().any():
            preds[d] = np.nan
            continue
        preds[d] = float(np.dot(np.r_[1.0, x_d.values], beta))

    return pd.Series(preds, name="har_oos_vol_annual").sort_index()

# -------- Pipeline --------
def run_pipeline(cfg: VolConfig, verbose: bool = False) -> Dict[str, Dict[str, pd.Series | pd.DataFrame]]:
    logging.info(f"Starting pipeline for tickers: {cfg.tickers}")
    prices = fetch_prices(cfg.tickers, cfg.start, cfg.end, cfg.price_col, cfg.data_dir)
    rets   = to_log_returns(prices)

    rows: list[dict] = []
    dm_rows: list[dict] = []
    out: Dict[str, Dict[str, pd.Series | pd.DataFrame]] = {}

    for t in cfg.tickers:
        logging.info(f"[{t}] ===== Starting processing =====")
        r = rets[t].dropna()
        if r.empty:
            logging.warning(f"[{t}] No returns data found, skipping.")
            continue

        # GARCH in-sample + single OOS point
        logging.info(f"[{t}] Fitting GARCH(1,1)...")
        sigma_in_pct, var_fcast_1d_pct2, vol_in_annual, vol_fcast_annual, res = fit_garch_and_forecast(r, cfg)
        logging.info(f"[{t}] GARCH fit done. {len(vol_in_annual)} obs.")
        rv = realized_vol(r, window=cfg.realized_window, annualization=cfg.annualization)
        logging.info(f"[{t}] Realized volatility computed ({cfg.realized_window}-day window).")
        
        # GARCH in-sample vs RV (baseline reference)
        rows.append({**accuracy_same_day(vol_in_annual, rv), "ticker": t, "model": "GARCH(1,1) (in-sample)"})

        # LSTM holdout (quick sanity check)
        try:
            logging.info(f"[{t}] Training simple LSTM holdout model...")
            lstm_pred, lstm_true, _ = fit_predict_lstm_rv(rv, cfg)
            rows.append({
                "ticker": t, "model": "LSTM (holdout test)",
                "RMSE": float(np.sqrt(mean_squared_error(lstm_true, lstm_pred))),
                "MAE":  float(mean_absolute_error(lstm_true, lstm_pred)),
                "MAPE(%)": float((np.abs((lstm_true - lstm_pred) / lstm_true).mean()) * 100),
                "R2":   float(r2_score(lstm_true, lstm_pred)),
                "n_obs": int(len(lstm_true)),
            })
        except Exception:
            logging.warning(f"[{t}] LSTM holdout failed: {e}")
            lstm_pred = pd.Series(dtype=float); lstm_true = pd.Series(dtype=float)
        logging.info(f"[{t}] LSTM holdout completed ({len(lstm_true)} test obs).")
        # LSTM walk-forward backtest (true OOS)
        try:
            logging.info(f"[{t}] Starting LSTM walk-forward backtest...")
            lstm_bt_pred, lstm_bt_true = backtest_lstm_full(
                rv_series=rv,
                cfg=cfg,
                min_obs=max(cfg.wf_min_obs, cfg.lstm_lookback + cfg.realized_window),
                retrain_every=cfg.wf_retrain_every,
                verbose=False
            )
            rows.append({**accuracy_same_day(lstm_bt_pred, lstm_bt_true), "ticker": t, "model": "LSTM Backtest (WF)"})
        except Exception:
            lstm_bt_pred = pd.Series(dtype=float); lstm_bt_true = pd.Series(dtype=float)
        logging.info(f"[{t}] LSTM backtest completed ({len(lstm_bt_pred)} OOS preds).")
        # GARCH OOS on same WF dates
        if not lstm_bt_true.empty:
            logging.info(f"[{t}] Running GARCH OOS forecasts...")
            garch_oos = oos_dynamic_vol(
                returns=r, target_dates=lstm_bt_true.index, cfg=cfg,
                vol_type="GARCH", p=1, o=0, q=1,
                min_obs=max(cfg.wf_min_obs, cfg.lstm_lookback + cfg.realized_window),
                verbose=False
            )
            rows.append({**accuracy_same_day(garch_oos, lstm_bt_true), "ticker": t, "model": "GARCH(1,1) OOS 1d (WF window)"})
        else:
            garch_oos = pd.Series(dtype=float)

        # HAR-RV OOS on same WF dates
        if not lstm_bt_true.empty:
            logging.info(f"[{t}] Running HAR-RV OOS forecasts...")
            har_oos = har_rv_oos(rv, lstm_bt_true.index, min_obs=60)
            rows.append({**accuracy_same_day(har_oos, lstm_bt_true), "ticker": t, "model": "HAR-RV OOS 1d (WF window)"})
        else:
            har_oos = pd.Series(dtype=float)

        # DM tests on aligned WF dates (MSE loss; lower is better)
        if not lstm_bt_true.empty:
            def add_dm(nameA, yA, nameB, yB):
                res_dm = dm_test(lstm_bt_true, yA, yB, h=1, power=2, lag=0, alternative="two-sided", harvey_adj=False)
                dm_rows.append({
                    "ticker": t, "A": nameA, "B": nameB,
                    "stat": res_dm["stat"], "p_value": res_dm["p_value"], "T": res_dm["T"]
                })

            if not garch_oos.empty:
                add_dm("LSTM WF", lstm_bt_pred, "GARCH OOS", garch_oos)
            if not har_oos.empty:
                add_dm("LSTM WF", lstm_bt_pred, "HAR-RV OOS", har_oos)
            if not (garch_oos.empty or har_oos.empty):
                add_dm("HAR-RV OOS", har_oos, "GARCH OOS", garch_oos)
        logging.info(f"[{t}] DM tests finished.")
        # Plot (kept, but lightweight; remove if you want absolute speed)
        try:
            logging.info(f"[{t}] Generating volatility comparison plot...")
            oos_dict = {
                "GARCH(1,1) OOS 1d (WF window)": garch_oos,
                "LSTM Backtest (WF)": lstm_bt_pred,
                "HAR-RV OOS 1d (WF window)": har_oos,
            }
            plot_vol_series_full(t, rv, vol_in_annual, vol_fcast_annual, oos_lines=oos_dict)
        except Exception:
            pass

        # Store artifacts
        out[t] = {
            "returns": r.rename("log_return"),
            "realized_vol_annual": rv,
            "garch_in_sample_vol_annual": vol_in_annual,
            "garch_forecast_vol_annual": vol_fcast_annual,
            "sigma_in_pct": sigma_in_pct,
            "var_forecast_1d_pct2": var_fcast_1d_pct2,
            "garch_params": res.params.to_frame(name="estimate"),
            "lstm_pred_vol_annual": lstm_pred,
            "lstm_true_vol_annual": lstm_true,
            "lstm_bt_pred_vol_annual": lstm_bt_pred,
            "lstm_bt_true_vol_annual": lstm_bt_true,
            "garch_oos_vol_annual": garch_oos,
            "har_oos_vol_annual": har_oos,
        }

    # Save tables
    
    acc_df = pd.DataFrame(rows, columns=["ticker","model","RMSE","MAE","MAPE(%)","R2","n_obs"])
    acc_df.to_csv("vol_accuracy_comparison.csv", index=False)
    if dm_rows:
        pd.DataFrame(dm_rows, columns=["ticker","A","B","stat","p_value","T"]).to_csv("dm_tests.csv", index=False)
    print("Saved: vol_accuracy_comparison.csv")
    if dm_rows: print("Saved: dm_tests.csv")
    logging.info(f"[{t}] Saved outputs and parameters CSVs.")
    return out



# ─────────────────────────── __main__ ───────────────────────────
if __name__ == "__main__":
    # Optional: make runs repeatable
    import random, os
    random.seed(0); np.random.seed(0)
    if torch.cuda.is_available():
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    else:
        torch.manual_seed(0)

    cfg = VolConfig(
        tickers=["AAPL", "MSFT"],
        start="2018-01-01",
        end=None,
        h_forecast=5,
        dist="t",
        realized_window=21,
        lstm_lookback=30, lstm_units=64, lstm_dropout=0.2,
        lstm_epochs=30, lstm_batch_size=32, lstm_train_frac=0.8, lstm_lr=1e-3
    )

    # Run the full pipeline (quiet training)
    results = run_pipeline(cfg, verbose=False)

    # Save per-ticker exports
    for t, d in results.items():
        parts = [
            d["garch_in_sample_vol_annual"],
            d["garch_forecast_vol_annual"],
            d["realized_vol_annual"],
            d.get("lstm_pred_vol_annual", pd.Series(dtype=float)),
            d.get("lstm_true_vol_annual", pd.Series(dtype=float)),
            d.get("lstm_bt_pred_vol_annual", pd.Series(dtype=float)),
            d.get("lstm_bt_true_vol_annual", pd.Series(dtype=float)),
            d.get("garch_oos_vol_annual", pd.Series(dtype=float)),
            d.get("har_oos_vol_annual", pd.Series(dtype=float)),
        ]
        export = pd.concat([s for s in parts if s is not None and not s.empty], axis=1)
        export.to_csv(f"{t}_vol_outputs.csv")
        d["garch_params"].to_csv(f"{t}_garch_params.csv")
        print(f"Saved: {t}_vol_outputs.csv")
        print(f"Saved: {t}_garch_params.csv")

    # Always print summary tables to console
    try:
        acc_df = pd.read_csv("vol_accuracy_comparison.csv")
        print("Saved: vol_accuracy_comparison.csv")
        print("\n=== Accuracy (OOS-aligned where applicable) ===")
        print(acc_df.sort_values(["ticker", "model"]).to_string(index=False))

        # Quick %-improvement lines per ticker
        for t in sorted(acc_df["ticker"].unique()):
            def pick(name):
                m = acc_df[(acc_df["ticker"] == t) & (acc_df["model"] == name)]
                return m.iloc[0] if not m.empty else None

            lwf  = pick("LSTM Backtest (WF)")
            goos = pick("GARCH(1,1) OOS 1d (WF window)")
            har  = pick("HAR-RV OOS 1d (WF window)")

            if lwf is not None and goos is not None and goos["RMSE"] > 0:
                imp = 100 * (1 - lwf["RMSE"] / goos["RMSE"])
                print(f"{t}: LSTM WF vs GARCH OOS — RMSE improvement {imp:.1f}%")
            if lwf is not None and har is not None and har["RMSE"] > 0:
                imp = 100 * (1 - lwf["RMSE"] / har["RMSE"])
                print(f"{t}: LSTM WF vs HAR OOS — RMSE improvement {imp:.1f}%")
    except Exception as e:
        print(f"[Summary Warning] {e}")

    # DM tests (if produced)
    try:
        if os.path.exists("dm_tests.csv"):
            dm_df = pd.read_csv("dm_tests.csv")
            print("Saved: dm_tests.csv")
            if not dm_df.empty:
                print("\n=== Diebold–Mariano tests (MSE loss) ===")
                dm_df_print = dm_df.copy()
                dm_df_print["p_value"] = dm_df_print["p_value"].map(lambda x: f"{x:.3g}")
                print(dm_df_print.to_string(index=False))
    except Exception as e:
        print(f"[DM Summary Warning] {e}")