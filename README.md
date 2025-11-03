# Volatility Forecasting (GARCH, HAR-RV, LSTM)

This project compares econometric and deep learning approaches for forecasting financial market volatility.  
Implemented models:
- **GARCH(1,1)**
- **HAR-RV**
- **LSTM (PyTorch, walk-forward retraining)**

### 🧠 Features
- Yahoo Finance data ingestion via `yfinance`
- Rolling realized volatility computation
- Out-of-sample walk-forward validation
- Diebold–Mariano statistical comparison
- CSV outputs for reproducibility

### 📊 Example Output
Plots and metrics are saved for each ticker (AAPL, MSFT).  
Models are compared via RMSE, MAE, MAPE, R², and Diebold–Mariano tests.

### 📁 File Structure
VOL-FORECASTING/
│
├── VolatilityForecasting.py # Main pipeline
├── vol_accuracy_comparison.csv # Model performance summary
├── dm_tests.csv # Diebold–Mariano test results
├── AAPL_vol_outputs.csv, MSFT_vol_outputs.csv
├── AAPL_garch_params.csv, MSFT_garch_params.csv
└── .gitignore
### Requirements


Python ≥ 3.9

PyTorch

ARCH

scikit-learn

yfinance

matplotlib

### Methodology Overview

1. GARCH(1,1) Model

The conditional variance evolves as:

𝜎
𝑡
2
=
𝜔
+
𝛼
𝜖
𝑡
−
1
2
+
𝛽
𝜎
𝑡
−
1
2
σ
t
2
    ​

=ω+αϵ
t−1
2
    ​

+βσ
t−1
2
    ​


where

𝜎
𝑡
2
σ
t
2
    ​

: conditional variance (volatility²)

𝜔
ω: long-run mean variance

𝛼
α: reaction to recent shocks

𝛽
β: persistence of past volatility


2. HAR-RV Model

The Heterogeneous AutoRegressive model for realized volatility:

𝑅
𝑉
𝑡
=
𝛽
0
+
𝛽
1
𝑅
𝑉
𝑡
−
1
+
𝛽
2
𝑅
𝑉
𝑡
−
5
:
𝑡
−
1
‾
+
𝛽
3
𝑅
𝑉
𝑡
−
22
:
𝑡
−
1
‾
+
𝜖
𝑡
RV
t
    ​

=β
0
    ​

+β
1
    ​

RV
t−1
    ​

+β
2
    ​

RV
t−5:t−1
    ​

    ​

+β
3
    ​

RV
t−22:t−1
    ​

    ​

+ϵ
t
    ​


It captures multi-horizon volatility components (daily, weekly, monthly).



3. LSTM Model

A recurrent neural network using Long Short-Term Memory cells to capture nonlinear temporal dependencies in volatility:

ℎ
𝑡
=
LSTM
(
𝑅
𝑉
𝑡
−
1
,
𝑅
𝑉
𝑡
−
2
,
…
,
𝑅
𝑉
𝑡
−
𝑝
)
h
t
    ​

=LSTM(RV
t−1
    ​

,RV
t−2
    ​

,…,RV
t−p
    ​

)

with hidden states trained via backpropagation through time (BPTT).
The model is retrained in a walk-forward fashion to mimic live trading conditions.

4. Diebold–Mariano Test

Used to compare forecast accuracy between two models 
𝐴
A and 
𝐵
B.

𝐷
𝑀
=
𝑑
ˉ
2
𝜋
𝑓
^
𝑑
(
0
)
𝑇
DM=
T
2π
f
^
    ​

d
    ​

(0)
    ​

    ​

d
ˉ
    ​


where 
𝑑
𝑡
=
𝐿
(
𝑒
𝐴
,
𝑡
)
−
𝐿
(
𝑒
𝐵
,
𝑡
)
d
t
    ​

=L(e
A,t
    ​

)−L(e
B,t
    ​

) and 
𝐿
L is squared error loss.
A significant negative statistic implies model A is more accurate.

### Sample Results (2018-2025)

Ticker    Model    RMSE    MAE    MAPE (%)    R²
AAPL    GARCH(1,1) (in-sample)    0.0583    0.0438    17.88    0.81
AAPL    LSTM (WF)    0.0345    0.0238    8.79    0.94
AAPL    HAR-RV (WF)    0.0212    0.0113    4.12    0.98
MSFT    GARCH(1,1) (in-sample)    0.0538    0.0398    17.34    0.82
MSFT    LSTM (WF)    0.0344    0.0229    9.56    0.93
MSFT    HAR-RV (WF)    0.0188    0.0102    4.01    0.98

Key takeaway:
Both LSTM and HAR-RV models achieve significant accuracy improvements over GARCH baselines, reducing RMSE by over 50% and explaining 93–98% of realized volatility variance.

### Diebold-Mariano Tests Results ( Forecast Accuracy Comparison)

Ticker    Comparison    DM Stat    p-value    Interpretation
AAPL    LSTM WF vs GARCH OOS    -9.54    < 0.001    LSTM significantly outperforms GARCH
AAPL    LSTM WF vs HAR-RV OOS    +12.13    < 0.001    HAR-RV significantly outperforms LSTM
AAPL    HAR-RV OOS vs GARCH OOS    -10.95    < 0.001    HAR-RV significantly outperforms GARCH
MSFT    LSTM WF vs GARCH OOS    -6.95    < 0.001    LSTM significantly outperforms GARCH
MSFT    LSTM WF vs HAR-RV OOS    +8.67    < 0.001    HAR-RV significantly outperforms LSTM
MSFT    HAR-RV OOS vs GARCH OOS    -7.95    < 0.001    HAR-RV significantly outperforms GARCH

### License

MIT License © Kevin Wood

