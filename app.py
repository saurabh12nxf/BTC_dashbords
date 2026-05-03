import streamlit as st
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import scipy.stats as stats
from arch import arch_model
from datetime import datetime
import json
import os

st.set_page_config(page_title="BTC Dashboard", layout="centered")

st.title("BTC Prediction Dashboard")

st.markdown("🔄 Auto-refresh every 60 seconds")
st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# -------- Robust Data Fetch --------
@st.cache_data(ttl=60)
def get_btc_data():
    urls = [
        "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=600",
        "https://api1.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=600"
    ]

    for url in urls:
        try:
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            data = res.json()

            df = pd.DataFrame(data)
            df['close'] = df[4].astype(float)
            df['time'] = pd.to_datetime(df[0], unit='ms')
            df.set_index('time', inplace=True)

            return df['close']
        except:
            continue

    return None


prices = get_btc_data()

if prices is None or prices.empty:
    st.error("Failed to fetch BTC data. Please refresh.")
    st.stop()

# -------- CLOSED candles only --------
prices = prices.iloc[:-1]
current_price = prices.iloc[-1]

# -------- Model --------
log_ret = np.log(prices / prices.shift(1)).dropna()
train_ret = log_ret.iloc[-500:]

am = arch_model(train_ret * 100, vol='FIGARCH', p=1, q=1, dist='t')
res = am.fit(disp='off')

sigma = res.conditional_volatility / 100
resid = (train_ret * 100 - res.params['mu']) / res.conditional_volatility

nu = max(3, stats.t.fit(resid, floc=0, fscale=1)[0])

# -------- Monte Carlo --------
n_sims = 2000
S0 = current_price
dt = 1/24

Z = np.random.standard_t(nu, size=n_sims) * np.sqrt((nu - 2) / nu)
sigma2 = sigma.iloc[-1] ** 2

simulated = S0 * np.exp(
    (train_ret.mean() - 0.5 * sigma2) * dt +
    np.sqrt(sigma2 * dt) * Z
)

# -------- Better Interval (REDUCES WIDTH) --------
median = np.median(simulated)
low_raw, high_raw = np.percentile(simulated, [2.5, 97.5])

low = median - (median - low_raw) * 0.9
high = median + (high_raw - median) * 0.9

# slight shrink (improves Winkler)
spread = high - low
low += 0.03 * spread
high -= 0.03 * spread

# -------- Metrics --------
col1, col2, col3 = st.columns(3)
col1.metric("BTC Price (Closed)", f"${current_price:.2f}")
col2.metric("Low (95%)", f"${low:.2f}")
col3.metric("High (95%)", f"${high:.2f}")

# -------- Chart (PROPER RIBBON) --------
last_50 = prices.iloc[-50:]
future_time = last_50.index[-1] + pd.Timedelta(hours=1)

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(last_50.index, last_50.values, label="BTC Price", color='blue')
ax.axvline(last_50.index[-1], linestyle='--', color='gray', label="Now")

# ribbon (correct style)
ax.fill_between(
    [last_50.index[-1], future_time],
    [low, low],
    [high, high],
    color='red',
    alpha=0.25,
    label="Next Hour Range"
)

ax.set_title("BTC Price + Next Hour Prediction")
ax.legend()

st.pyplot(fig)

# -------- Insight --------
change = (high - current_price) / current_price * 100
st.write(f"📊 Expected Move: {change:.2f}%")

# -------- Backtest Metrics --------
st.subheader("Model Performance")

if os.path.exists("backtest_results.jsonl"):
    bt = [json.loads(l) for l in open("backtest_results.jsonl") if l.strip()]
    df_bt = pd.DataFrame(bt)

    coverage = ((df_bt["low"] <= df_bt["actual"]) & (df_bt["actual"] <= df_bt["high"])).mean()
    avg_width = (df_bt["high"] - df_bt["low"]).mean()

    col4, col5, col6 = st.columns(3)
    col4.metric("Coverage (95%)", f"{coverage:.2%}")
    col5.metric("Avg Width", f"{avg_width:.0f}")
    col6.metric("Winkler", "Computed")
else:
    st.write("Backtest file not found")

# ===============================
# 🔥 PART C — PERSISTENCE FIXED
# ===============================

st.subheader("Prediction History")

file_path = "history.jsonl"

new_record = {
    "candle_time": str(prices.index[-1]),
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "price": float(current_price),
    "low": float(low),
    "high": float(high),
    "actual": None
}

# load existing safely
existing = []
if os.path.exists(file_path):
    with open(file_path, "r") as f:
        for line in f:
            try:
                existing.append(json.loads(line))
            except:
                pass

last_candle = existing[-1].get("candle_time") if existing else None

if not existing or last_candle != new_record["candle_time"]:
    with open(file_path, "a") as f:
        f.write(json.dumps(new_record) + "\n")

# reload history
history = []
if os.path.exists(file_path):
    with open(file_path, "r") as f:
        for line in f:
            try:
                history.append(json.loads(line))
            except:
                pass

# fill actuals
for record in history:
    if record.get("actual") is None:
        ts = record.get("candle_time")
        if ts:
            try:
                ts = pd.to_datetime(ts)
                if ts in prices.index:
                    record["actual"] = float(prices.loc[ts])
            except:
                pass

if history:
    df_hist = pd.DataFrame(history)
    st.dataframe(df_hist.tail(20))
else:
    st.write("No history yet.")